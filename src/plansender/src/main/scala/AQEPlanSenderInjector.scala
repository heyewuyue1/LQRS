package xxxxxxxxxx

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSessionExtensions

class AQEPlanSenderInjector extends (SparkSessionExtensions => Unit) with Logging{
  override def apply(extensions: SparkSessionExtensions): Unit = {
      val lqrsEnabled = System.getProperty("spark.sql.adaptive.lqrs.enabled", "false")
      if (lqrsEnabled == "true") {
        extensions.injectRuntimeOptimizerRule(session => {
          new AQEPlanSender(session, true)
        })
        extensions.injectOptimizerRule(session => {
          new AQEPlanSender(session, false)
        })
        logWarning("AQEPlanSender is injected")
      }
    }
}
