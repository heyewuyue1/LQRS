package xxxxxxxxxx

import org.apache.log4j.LogManager
import org.apache.spark.SparkConf
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{Cost, CostEvaluator, SimpleCost}
import org.apache.spark.sql.execution.exchange.ShuffleExchangeLike

object ExtraCost {
  var numShuffles: Int = 0
}

case class EqualCostEvaluator(conf: SparkConf) extends CostEvaluator {
  private val logger =  LogManager.getLogger(this.getClass)
  override def evaluateCost(plan: SparkPlan): Cost = {
    ExtraCost.numShuffles = plan.collect {
      case s: ShuffleExchangeLike => s
    }.size

    logger.warn(s"Evaluating cost for:\n$plan")
    SimpleCost(0L)
  }
}
