package xxxxxxxxxx

import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.QueryExecution
import org.apache.spark.sql.util.QueryExecutionListener

class QueryRegisterListener extends QueryExecutionListener with Logging {
  private val serverUri = System.getProperty("spark.sql.adaptive.lqrs.uri", "http://localhost:10533")
  private var lastExecutionTime = -1L
  private val webUtils = new WebUtils(serverUri)

  override def onSuccess(funcName: String, qe: QueryExecution, durationNs: Long): Unit = {
    logWarning(s"durationNs: $durationNs")
    lastExecutionTime = durationNs
    val sessionName = qe.sparkSession.sparkContext.getConf.get("spark.app.name")
    webUtils.sendRegisterRequest(sessionName, qe.executedPlan.toString, durationNs, OptimizationCost.CPlan)
    OptimizationCost.CPlan = 0
  }

  override def onFailure(funcName: String, qe: QueryExecution, exception: Exception): Unit = {
    logError(s"Query execution failed: ${exception.getMessage}")
    lastExecutionTime = -lastExecutionTime - 3
    val sessionName = qe.sparkSession.sparkContext.getConf.get("spark.app.name")
    webUtils.sendRegisterRequest(sessionName, qe.executedPlan.toString, lastExecutionTime, OptimizationCost.CPlan)
    OptimizationCost.CPlan = 0
  }
}
