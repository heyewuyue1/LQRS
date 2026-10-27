package xxxxxxxxxx

import sttp.client4.quick._
import sttp.model.HttpVersion.HTTP_1_1
import sttp.model.StatusCode.Ok
import upickle.default._
import org.apache.log4j.{Logger, LogManager}

case class PlanReqBody(plan: String,
                      executedRowCounts: Seq[(String, (Boolean, String, String, Set[String]))],
                      runTime: Boolean,
                      triggerTime: Int,
                      numShuffles: Int)
case class RegisterReqBody(sessionName: String, finalPlan: String, executionTime: Long, CPlan: Long)


case class AQEResponse(actionType: String, action: List[Int], increaseMOT: Int)
object AQEResponse {
  implicit val rw: ReadWriter[AQEResponse] = macroRW
}

class WebUtils(serverUri: String) {
  private val logger =  LogManager.getLogger(this.getClass)

  def sendPlanRequest(
                       plan: String,
                       executedRowCounts: Seq[(String, (Boolean, String, String, Set[String]))],
                       runTime: Boolean,
                       triggerTime: Int,
                       numShuffles: Int): (String, List[Int]) = {
    val reqBody = PlanReqBody(plan, executedRowCounts, runTime, triggerTime, numShuffles)
    implicit val reqBodyRW: ReadWriter[PlanReqBody] = macroRW
    val reqBodyStr: String = write(reqBody)
    logger.warn(reqBodyStr)
    logger.warn(s"Sending plan: $plan")
    try {
      val response = quickRequest
        .post(uri"$serverUri/plan")
        .header("Content-Type", "application/json")
        .httpVersion(HTTP_1_1)
        .body(reqBodyStr)
        .send()

      if (response.code == Ok) {
        logger.info("Plan request sent successfully")
        logger.warn("Get response OK")
        logger.warn(response.body)
        val parsed = read[AQEResponse](response.body)
        val actionType = parsed.actionType
        val action = parsed.action
        return (actionType, action)
      } else {
        logger.error(s"Failed to send plan request: ${response.code}")
      }
    } catch {
      case e: Exception => logger.error("Error sending plan request", e)
    }
    ("no-op", List())
  }

  def sendRegisterRequest(sessionName: String, finalPlan: String, executionTime: Long, CPlan: Long): Unit = {
    val reqBody = RegisterReqBody(sessionName, finalPlan, executionTime, CPlan)
    implicit val ownerReg: ReadWriter[RegisterReqBody] = macroRW
    val registerBodyStr: String = write(reqBody)
    try {
      quickRequest
        .post(uri"$serverUri/register")
        .header("Content-Type", "application/json")
        .httpVersion(HTTP_1_1)
        .body(registerBodyStr)
        .send()
      logger.warn("Query execution registered")
    } catch {
      case e: Exception => logger.error("Error register query execution", e)
    }
  }
}
