from llm.wrapper import LLMWrapper
import pandas as pd 
from agent.utils import summarize_value, resolve_args_from_scratchpad
from agent.scratchpad import Scratchpad
from agent.context_history import ContextHistory
import streamlit as st

class ReActPlanExecutor:
    def __init__(self, tool_specs, tool_mapper, llm: LLMWrapper, use_ui= True, max_retries: int = 5):
        self.tools = tool_specs                # Toolset for actions (e.g., sql_tool, ml_tool, plot_tool)
        self.tool_mapper = tool_mapper          # Maps tool names to functions
        self.llm = llm                    # LLM interface
        self.context_history = ContextHistory()        # Global trace of steps
        self.scratchpad = Scratchpad()             # Volatile working memory
        self.current_step_index = 0      # Pointer to step in the plan
        self.max_tries = max_retries      # Max retries for each step
        self.use_ui = use_ui

    def reset(self):
        self.context_history.clear()
        self.scratchpad.clear()
        self.current_step_index = 0

    def run_plan(self, plan: list[str], question: str):
        self.status_items = []

        while self.current_step_index < len(plan):
            step = plan[self.current_step_index]
            step_label = f"Running step {self.current_step_index + 1}: {step}"
            status = None

            try:
                if self.use_ui:
                    status = st.status(step_label, state="running", expanded=True)
                    self.status_items.append(status)
                    status.write(f"**Step {self.current_step_index + 1}:** {step}")

                print(f"\n--- Step {self.current_step_index + 1}: {step} ---")
                self.execute_step(step, question)

                latest_entry = self.context_history.entries[-1]["trace"]
                if latest_entry:
                    last = latest_entry[-1]
                    if self.use_ui and status:
                        status.write(f"- **Thought:** {last.get('thought', '')}")
                        status.write(f"- **Action:** {last.get('action', '')}")
                        status.write(f"- **Observation:** {last.get('observation', '')}")
                if self.use_ui and status:
                    status.update(label=f"✅ Completed: {step}", state="complete", expanded=False)

            except Exception as e:
                error_msg = f"❌ Step failed due to: {str(e)}"
                print(error_msg)
                if self.use_ui and status:
                    status.write(f"**Error:** {str(e)}")
                    status.update(label=error_msg, state="error", expanded=True)
                break

            self.current_step_index += 1

        # self._display_final_output()
        final_prompt = self._format_context(question)
        return self.llm._call_llm(final_prompt)

    def execute_step(self, step: str, question: str):
        step_done = False
        curr_tries = 0
        local_trace = []  # For this step’s ReAct loop

        while not step_done and curr_tries <= self.max_tries:
            curr_tries += 1

            # STEP 1: THINK
            prompt = self._build_prompt(question, step, local_trace)
            thought_output = self.llm.think_and_route(prompt)
            print(thought_output)
            local_trace.append({"thought": thought_output.get('thought', 'There was an error!')})
            
            # STEP 2: ACT
            if thought_output.get('tool') and thought_output.get('tool') != 'think_reflect':
                action = {
                    "tool": thought_output["tool"],
                    "args": thought_output["args"],
                }
                result = self._execute_action(action)
                local_trace[-1]["action"] = action
                local_trace[-1]["observation"] = summarize_value(result)
            else:
                local_trace[-1]["action"] = "None"
                local_trace[-1]["observation"] = thought_output.get('thought')
            # STEP 3: OBSERVE + DECIDE
            step_done = self._is_step_complete(step, local_trace)
            if step_done:
                print("✅ Step complete.")

        if not step_done:
            print("❌ Step failed after max tries.")
            local_trace.append({
                "thought": "Exceeded max retries.",
                "action": "None",
                "observation": "Step aborted after max attempts."
            })

        # Add to global context history
        self.context_history.log(step, local_trace)


    def _build_prompt(self, question, step, trace):
        return {
            "question": question, # user question
            "step_description": step, # current step
            "scratchpad": self.scratchpad, # scratchpad of accessible variables
            "recent_trace": trace[-3:]  # Truncate local trace for token efficiency
        }

    def _execute_action(self, action: dict):
        tool_name = action.get("tool")
        tool_args = action.get("args", {})

        # Extract output_var (must be present per spec)
        output_var = tool_args.pop("output_var", None)
        if not output_var:
            raise ValueError(f"Tool '{tool_name}' must specify 'output_var' to store results.")

        args = resolve_args_from_scratchpad(tool_args, self.scratchpad)
        # Execute the tool
        result = self.tool_mapper[tool_name](**args)

        # Store the result in the scratchpad under the provided variable name
        self.scratchpad.set(output_var, result)
        self.scratchpad.set("_last_output_var", output_var)

        return result

    def _is_step_complete(self, step, trace):
        return self.llm.judge_step(step, trace)
    
    def _format_context(self, question):
        prompt_lines = ["You are a data scientist that has completed the following steps:\n"]
        print(self.context_history.entries)
        for idx, entry in enumerate(self.context_history.entries):
            step = entry["step"]
            prompt_lines.append(f"Step {idx + 1}: {step}")

            for t in entry["trace"]:
                thought = t.get("thought", "")
                action = t.get("action", "")
                observation = t.get("observation", "")

                if thought:
                    prompt_lines.append(f"Thought: {thought}")
                if action:
                    prompt_lines.append(f"Action: {action}")
                if observation:
                    prompt_lines.append(f"Observation: {observation}")
                prompt_lines.append("\n")  # for spacing

        if '_last_output_var' in self.scratchpad:
            output_var = self.scratchpad.get("_last_output_var")
            result = self.scratchpad.get(output_var)

            if isinstance(result, pd.DataFrame):
                preview = result.head(10).to_markdown(index=False)
                prompt_lines.append(f"Here is the final result from `{output_var}` (DataFrame preview):\n")
                prompt_lines.append(preview)
            elif isinstance(result, str):
                prompt_lines.append(f"Here is the final string result from `{output_var}`:\n{result}")
                prompt_lines.append(f"Based on the above, answer the following: {question.strip()}")
        return "\n".join(prompt_lines)
    
    # def _display_final_output(self):
    #     try:
    #         output_var = self.scratchpad.get("_last_output_var")
    #         final_result = self.scratchpad.get(output_var)
    #     except Exception:
    #         return

    #     st.markdown(f"### 🧾 Final Output: `{output_var}`")

    #     if isinstance(final_result, pd.DataFrame):
    #         st.dataframe(final_result)
    #     elif isinstance(final_result, str):
    #         st.code(final_result)
    #     else:
    #         try:
    #             st.write(final_result)
    #         except Exception:
    #             st.warning("Unable to display final result.")


