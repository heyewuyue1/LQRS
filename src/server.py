from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from utils.logger import setup_custom_logger
from config import ServerConfig
from lqrs import LQRS

app = FastAPI()
logger = setup_custom_logger("SERVER")

if ServerConfig.mode == "train":
    agent = LQRS(deterministic=False)
elif ServerConfig.mode == "test":
    agent = LQRS(deterministic=True)

class PlanRequest(BaseModel):
    plan: str
    executedRowCounts: list
    runTime: bool
    triggerTime: int
    numShuffles: int

class RegisterRequest(BaseModel):
    sessionName: str
    finalPlan: str
    executionTime: int
    CPlan: int

@app.post("/plan")
async def receive_plan(request: PlanRequest):
    plan = request.plan
    executed_row_counts = request.executedRowCounts
    executed_row_counts = dict(executed_row_counts)
    runtime = request.runTime
    trigger_time = request.triggerTime
    num_shuffles = request.numShuffles
    action = {
        "actionType": 'no-op',
        "action": ()
    }
    action["actionType"], action["action"] = agent.act(plan, executed_row_counts, runtime, trigger_time, num_shuffles)
    return action

@app.post("/register")
async def receive_plan(request: RegisterRequest):
    finalPlan = request.finalPlan
    sessionName = request.sessionName
    execution_time = request.executionTime
    plan_time = request.CPlan
    agent.replay_if_can(execution_time, plan_time)
    logger.debug(f"Final Plan of {sessionName}: \n {finalPlan}")
    return {}

if __name__ == "__main__":
    uvicorn.run("server:app", host="localhost", port=10533)
