package xxxxxxxxxx
import org.apache.spark.sql.catalyst.plans.logical.{BROADCAST, HintInfo, Join, JoinHint, LogicalPlan, Project, UnaryNode}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{And, Attribute, AttributeSet, ExpressionSet, PredicateHelper}
import org.apache.spark.sql.catalyst.optimizer.Cost
import org.apache.spark.sql.catalyst.optimizer.JoinReorderDP.JoinPlan
import org.apache.spark.sql.execution.adaptive.{LogicalQueryStage, QueryStageExec}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.rules.Rule

object OptimizationCost {
  var CPlan = 0L
}

class AQEPlanSender(session: SparkSession, runTime: Boolean) extends Rule[LogicalPlan] with Logging with PredicateHelper {

  private val maxOperateTime =
    if (runTime) System.getProperty("spark.sql.adaptive.lqrs.maxOperateTime", "3").toInt
    else System.getProperty("spark.sql.lqrs.maxOperateTime", "3").toInt
  private val serverUri = System.getProperty("spark.sql.adaptive.lqrs.uri", "http://localhost:10533")
  private val triggerInterval = System.getProperty("spark.sql.lqrs.triggerInterval", "2").toInt
  private var triggerTime = 0 // trigger first time
  private var curOperateCnt = 0
  private val webUtils = new WebUtils(serverUri)

  override def apply(plan: LogicalPlan): LogicalPlan = {

    if (serverUri == "") {
      logError("spark.sql.adaptive.lqrs.uri is not set")
      return plan
    }
    logWarning(s"runTime: $runTime triggerTime: $triggerTime, curOperateCnt: $curOperateCnt, maxOperateTime: $maxOperateTime")
    if (triggerTime % triggerInterval != 0) {
      triggerTime += 1
      return plan
    }
    triggerTime += 1
    if (curOperateCnt == maxOperateTime) {
      logWarning("Max operate time reached.")
      return plan
    }
    if (plan.toString.split("\n").length < 3) {
      logWarning("Plan is too small")
      return plan
    }
    val start = System.currentTimeMillis()
    var executedStages: Seq[(String, (Boolean,String, String, Set[String]))] = Seq()
    if (runTime) {
      executedStages = plan.collect {
        case lqs @ LogicalQueryStage(_, stage: QueryStageExec) =>
          val tablePattern = """FileScan parquet spark_catalog\.[a-zA-Z0-9_]+\.(\w+)""".r
          val tables: Set[String] = tablePattern
            .findAllMatchIn(stage.plan.toString)
            .map(_.group(1))
            .toSet
          lqs.toString.trim -> (
            stage.isMaterialized,
            stage.getRuntimeStatistics.rowCount.getOrElse(-1L).toString, // rowCount
            stage.getRuntimeStatistics.sizeInBytes.toString,             // sizeInBytes
            tables
          )
      }
    }

    val (actionType, action) =
        webUtils.sendPlanRequest(plan.toString, executedStages, runTime, triggerTime, ExtraCost.numShuffles)
    if (actionType != "no-op") {
      curOperateCnt += 1
      if (actionType == "lead") {
        logWarning(s"Lead$action")
        val newPlan = reorderTopLevelJoin(plan, action)
        logWarning(s"New plan: $newPlan")
        OptimizationCost.CPlan += (System.currentTimeMillis() - start)
        logWarning(s"start: $start, end: ${System.currentTimeMillis()}, cost: ${OptimizationCost.CPlan}")
        return newPlan
      } else if (actionType == "swap") {
        logWarning(s"Swap$action")
        val newPlan = reorderTopLevelJoin(plan, action)
        logWarning(s"New plan: $newPlan")
        OptimizationCost.CPlan += (System.currentTimeMillis() - start)
        return newPlan
      } else if (actionType == "hint") {
        logWarning(s"Hint$action")
        val newPlan = hintPlan(plan, action)
        logWarning(s"New plan: $newPlan")
        OptimizationCost.CPlan += (System.currentTimeMillis() - start)
        return newPlan
      } else if (actionType == "cbo") {
        if (!runTime) {
          if (action.head == 0) {
            logWarning("Disable CBO")
            session.conf.set("spark.sql.cbo.enabled", "false")
            session.conf.set("spark.sql.cbo.joinReorder.enabled", "false")
          } else {
            logWarning("Enable CBO")
            session.conf.set("spark.sql.cbo.enabled", "true")
            session.conf.set("spark.sql.cbo.joinReorder.enabled", "true")
            session.conf.set("spark.sql.cbo.joinReorder.dp.threshold", "18")
          }
        } else {
          logWarning("Receiving cbo actions while running (which should not happen).")
          curOperateCnt -= 1
        }
      }
    }
    else logWarning("No-op")
    OptimizationCost.CPlan += (System.currentTimeMillis() - start)
    plan
  }

  private def reorderTopLevelJoin(plan: LogicalPlan, action: List[Int]): LogicalPlan = {
    def rewrite(plan: LogicalPlan): LogicalPlan = plan match {
      case join: Join =>
        if (action.length == 1) {
          reorderJoinLeading(join, action)
        } else {
          reorderJoinSwapping(join, action)
        }
      case u: UnaryNode =>
        u.withNewChildren(Seq(rewrite(u.child)))
      case other =>
        other
    }
    rewrite(plan)
  }

  private def reorderJoinLeading(plan: LogicalPlan, leading: List[Int]): LogicalPlan = {
    val (innerJoins, conditions) = extractInnerJoins(plan)
    logWarning(s"Inner joins: ${innerJoins.mkString(", ")}".replace("\n", ""))
    logWarning(s"Conditions: ${conditions.mkString(", ")}".replace("\n", ""))
    if (innerJoins.length < 2) {
      logWarning("No need to reorder join")
      return plan
    }
    val indexSet = leading.toSet
    val front = leading.map { idx =>
      if (idx < innerJoins.length) {
        innerJoins(idx)
      } else {
        logWarning(s"leading index $idx out of range")
        return plan
      }
    }
    val rest = innerJoins.zipWithIndex.filterNot { case (_, idx) => indexSet.contains(idx) }.map(_._1)
    var reordered = front ++ rest
    logWarning(s"Initial reordered: $reordered") 

    var currentPlan = JoinPlan(Set(), reordered.head, ExpressionSet(), Cost(0, 0))
    reordered = reordered.patch(0, Nil, 1)

    while (reordered.nonEmpty) {
      var i = 0
      var joined = false
      while (i < reordered.length && !joined) {
        val candidate = reordered(i)
        val candidatePlan = JoinPlan(Set(), candidate, ExpressionSet(), Cost(0, 0))
        buildJoin(currentPlan, candidatePlan, conditions, plan.outputSet) match {
          case Some(p) =>
            currentPlan = p
            reordered = reordered.patch(i, Nil, 1)
            joined = true
            logWarning(s"Build new join plan: $currentPlan")
            logWarning(s"Reordered: $reordered")
          case None =>
            i += 1
        }
      }

      if (!joined) {
        logWarning("Cannot find joinable plan from reordered (which should not happen), return original plan")
        return plan
      }
    }
    currentPlan.plan
  }

  private def reorderJoinSwapping(plan: LogicalPlan, swapping: List[Int]): LogicalPlan = {
    val (innerJoins, conditions) = extractInnerJoins(plan)
    logWarning(s"Inner joins: ${innerJoins.mkString(", ")}".replace("\n", ""))
    logWarning(s"Conditions: ${conditions.mkString(", ")}".replace("\n", ""))
    val elemI = innerJoins(swapping(0))
    val elemJ = innerJoins(swapping(1))
    var reordered = innerJoins.updated(swapping(0), elemJ).updated(swapping(1), elemI)


    logWarning(s"Initial reordered: $reordered") 

    var currentPlan: Option[JoinPlan] = Some(JoinPlan(Set(), reordered.head, ExpressionSet(), Cost(0, 0)))
    reordered = reordered.patch(0, Nil, 1)
    while (currentPlan.isDefined && reordered.nonEmpty) {
      val candidate = reordered(0)
      val candidatePlan = JoinPlan(Set(), candidate, ExpressionSet(), Cost(0, 0))
      currentPlan =  buildJoin(currentPlan.get, candidatePlan, conditions, plan.outputSet)
      if (!currentPlan.isDefined) {
        logWarning("Cannot find joinable plan from reordered, return original plan")
        return plan
      }
      reordered = reordered.patch(0, Nil, 1)
      logWarning(s"Build new join plan: $currentPlan")
      logWarning(s"Reordered: $reordered")
    }
    currentPlan.get.plan
  }


  private def extractInnerJoins(plan: LogicalPlan): (Seq[LogicalPlan], ExpressionSet) = {
    plan match {
      case Join(left, right, _, Some(cond), _) =>
        val (leftPlans, leftConditions) = extractInnerJoins(left)
        val (rightPlans, rightConditions) = extractInnerJoins(right)
        (leftPlans ++ rightPlans, leftConditions ++ rightConditions ++
          splitConjunctivePredicates(cond))
      case Project(projectList, j @ Join(_, _, _, _, _))
        if projectList.forall(_.isInstanceOf[Attribute]) =>
        extractInnerJoins(j)
      case _ =>
        (Seq(plan), ExpressionSet())
    }
  }

  /**
   * Builds a new JoinPlan if the following conditions hold:
   * - the sets of items contained in left and right sides do not overlap.
   * - there exists at least one join condition involving references from both sides.
   *
   * @param oneJoinPlan One side JoinPlan for building a new JoinPlan.
   * @param otherJoinPlan The other side JoinPlan for building a new join node.
   * @param conf SQLConf for statistics computation.
   * @param conditions The overall set of join conditions.
   * @param topOutput The output attributes of the final plan.
   * @param filters Join graph info to be used as filters by the search algorithm.
   * @return Builds and returns a new JoinPlan if both conditions hold. Otherwise, returns None.
   */
  private def buildJoin(
                         oneJoinPlan: JoinPlan,
                         otherJoinPlan: JoinPlan,
                         conditions: ExpressionSet,
                         topOutput: AttributeSet): Option[JoinPlan] = {

    if (oneJoinPlan.itemIds.intersect(otherJoinPlan.itemIds).nonEmpty) {
      // Should not join two overlapping item sets.
      return None
    }

    val onePlan = oneJoinPlan.plan
    val otherPlan = otherJoinPlan.plan
    val joinConds = conditions
      .filterNot(l => canEvaluate(l, onePlan))
      .filterNot(r => canEvaluate(r, otherPlan))
      .filter(e => e.references.subsetOf(onePlan.outputSet ++ otherPlan.outputSet))
    if (joinConds.isEmpty) {
      // Cartesian product is very expensive, so we exclude them from candidate plans.
      // This also significantly reduces the search space.
      logWarning(s"Cartesian product detected between $onePlan and $otherPlan, skipping this join.")
      return None
    }

    // Put the deeper side on the left, tend to build a left-deep tree.
    val (left, right) = if (oneJoinPlan.itemIds.size >= otherJoinPlan.itemIds.size) {
      (onePlan, otherPlan)
    } else {
      (otherPlan, onePlan)
    }
    val newJoin = Join(left, right, Inner, joinConds.reduceOption(And), JoinHint.NONE)
    val collectedJoinConds = joinConds ++ oneJoinPlan.joinConds ++ otherJoinPlan.joinConds
    logWarning(s"collectedJoinConds: $collectedJoinConds")
    val remainingConds = conditions -- collectedJoinConds
    logWarning(s"remainingConds: $remainingConds")
    val neededAttr = AttributeSet(remainingConds.flatMap(_.references)) ++ topOutput
    val neededFromNewJoin = newJoin.output.filter(neededAttr.contains)
    logWarning(s"neededFromNewJoin: $neededFromNewJoin")
    val newPlan =
      if ((newJoin.outputSet -- neededFromNewJoin).nonEmpty) {
        Project(neededFromNewJoin, newJoin)
      } else {
        newJoin
      }

    val itemIds = oneJoinPlan.itemIds.union(otherJoinPlan.itemIds)
    // Now the root node of onePlan/otherPlan becomes an intermediate join (if it's a non-leaf
    // item), so the cost of the new join should also include its own cost.
    val newPlanCost = oneJoinPlan.planCost + otherJoinPlan.planCost
    Some(JoinPlan(itemIds, newPlan, collectedJoinConds, newPlanCost))
  }



  private def hintPlan(plan: LogicalPlan, hintList: List[Int]): LogicalPlan = {
    val (targetJoinIndex, hintSide) =
      if (hintList.head == 0) (0, 0)
      else (hintList.head - 1, 1)

    var joinCnt = 0

    plan.transformUp {
      case j @ Join(left, right, joinType, condition, hint) if joinCnt == targetJoinIndex =>
        val newHint = hintSide match {
          case 0 => hint.copy(leftHint =
            Some(hint.leftHint.getOrElse(HintInfo()).copy(strategy = Some(BROADCAST))))
          case 1 => hint.copy(rightHint =
            Some(hint.rightHint.getOrElse(HintInfo()).copy(strategy = Some(BROADCAST))))
          case _ => hint
        }
        logWarning("Hint set to BROADCAST")
        joinCnt += 1
        j.copy(hint = newHint)
      case j @ Join(left, right, joinType, condition, hint) =>
        joinCnt += 1
        j
    }

  }
}
