from llm.wrapper import LLMWrapper
from typing import List, Dict, Any

class Planner:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def create_plan(self, question: str, context: str = "") -> List[str]:
        """
        Generates a plan from a business question using the LLM.
        Returns a list of natural-language steps.
        """
        return self.llm.plan(question=question, context=context)
