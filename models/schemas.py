from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AgentRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class StepTrace(BaseModel):
    thought: str
    action: str
    observation: str

class StepResult(BaseModel):
    step: str
    trace: List[StepTrace]

class AgentResponse(BaseModel):
    plan: List[str]
    execution_trace: List[StepResult]
    final_summary: str
