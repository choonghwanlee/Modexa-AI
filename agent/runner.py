from agent.planner import Planner
from agent.executor import ReActPlanExecutor
from agent.tools import tool_specs, tool_mapper
from llm.wrapper import LLMWrapper

def run_agent_pipeline(question, use_ui=True):
    llm = LLMWrapper(tool_specs=tool_specs)
    planner = Planner(llm)
    executor = ReActPlanExecutor(
        tool_specs=tool_specs,
        tool_mapper=tool_mapper,
        llm=llm,
        use_ui=use_ui,
    )

    plan = planner.create_plan(question)
    return plan, executor.run_plan(plan, question)
