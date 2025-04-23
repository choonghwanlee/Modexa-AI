from openai import OpenAI
from typing import List, Dict, Any
import json
import inspect
from llm.prompts import dbschema_str
from dotenv import load_dotenv
import re
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   

client = OpenAI()

class LLMWrapper:
    def __init__(self, model="gpt-4.1", temperature=0.3, tool_specs: List[Dict] = None):
        self.model = model
        self.temperature = temperature
        self.tool_specs = tool_specs or []

    def _call_llm(self, prompt: str):
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            'type': "input_text",
                            "text": prompt
                        }
                    ]
                },
            ],
            temperature=self.temperature,
        )
        return response.output_text

    def think_and_route(self, context: Dict[str, Any]) -> Dict:
        """
        Calls OpenAI with function-calling and routes to a tool.
        
        """
        question = context["question"]
        step = context["step_description"]
        trace = context.get("recent_trace", [])
        scratchpad = context.get("scratchpad", {})

        history_str = "\n".join([
            f"Thought: {t['thought']}\nAction: {t['action']}\nObservation: {t['observation']}"
            for t in trace
        ])

        scratchpad_str = "\n".join([f"{k}: {v}" for k, v in scratchpad.describe().items()])

        prompt = (
            f"You are an enterprise data scientist helping a user answer a business problem: {question}. You have devised a step-by-step plan for answering this question, and currently working on tackling this step: '{step}'\n\n"
            f"Recent trace of actions & observations:\n{history_str}\n\n"
            f"Scratchpad of variables we can reference in memory:\n{scratchpad_str}\n\n"
            f"Here is the database schema we can write SQL queries to fetch data from:\n{dbschema_str}\n\n"
            f"What should you do next? First, reason about the next action to take based on what you've observed. Finally, decide on what tool to call. Do not make up new models or data columns that do not exist"
            f"If no tool is appropriate, use the `think_reflect` tool with a note that explains your thought process. If you're using the 'write_python_code' function, pass input parameters using a single scratchpad variable or a comma-separated list of variable names."
        )

        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user", 
                    "content": [
                        {
                            'type': "input_text",
                            "text": prompt
                        }
                    ]
                },
            ],
            temperature=self.temperature,
            tools=self.tool_specs,
            tool_choice="auto",
        )
        results = self._parse_thought(response)
        return results
    

    def plan(self, question: str, context: str = "") -> List[str]:
        prompt = (
            f"You are a seasoned data scientist helping the user answer the following business question for their company: {question}. You have access to a set of models % Tools and a database % Database to help you answer the question.\n\n"
            "% Task:\nFor the given business question, generate a step-by-step plan for the data and tools to use for the task. This plan should involve individual tasks, that if executed correctly, will generate the information you need to answer the question. Do not add any superfluous steps, and prioritize being as concise as possible. This includes minimizing calls to `write_python_code` and fetching and manipulating data mostly via `convert_text_to_sql`. Make sure the each step in the plan is grounded in the tools and data we are provided with – do not make up new models or columns.\n\n"
            f"% Tools:\n{self._summarize_toolspecs(self.tool_specs)}\n\n"
            f"% Database:\n{dbschema_str}"
            "% Output Format:\nThink step by step about how to break down the question into smaller tasks. Finally, generate a numbered list for each step in the plan under a ## Final Plan header. Make sure each step is in one line. Include a final step to make sure the final output variable is in a presentable format (code/markdown or table, graph)"
        )
        if context:
            prompt += f"Additional Context:\n{context}\n\n"

        response = self._call_llm(prompt)
        print(response)
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> List[str]:
        # Locate the start of the Final Plan section
        match = re.search(r"## Final Plan\s*\n", response, re.IGNORECASE)
        if not match:
            return []

        # Slice from the start of Final Plan
        plan_text = response[match.end():]

        # Extract numbered steps using regex
        steps = re.findall(r"^\s*\d+\.\s*(.+)", plan_text, re.MULTILINE)

        return steps

    def _summarize_toolspecs(self, tool_specs):
        summary = []
        for tool in tool_specs:
            name = tool.get("name", "Unnamed tool")
            desc = tool.get("description", "No description provided.")
            params = tool.get("parameters", {})
            props = params.get("properties", {})
            required = set(params.get("required", []))

            tool_summary = [f"### Tool: `{name}`", f"**Description**: {desc}", "**Parameters:**"]

            if not props:
                tool_summary.append("*(No parameters)*")
            else:
                for param_name, param_info in props.items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "No description.")
                    required_flag = " (required)" if param_name in required else ""
                    tool_summary.append(f"- `{param_name}`: `{param_type}`{required_flag} — {param_desc}")

            summary.append("\n".join(tool_summary))

        return "\n\n".join(summary)

    def _parse_thought(self, response):
        """
        Parses an OpenAI function-calling response, extracting both
        - The reasoning text ("thought")
        - The first function call (tool + args)
        """
        results = {
            "thought": None,
            "tool": None,
            "args": None
        }

        for entry in response.output:
            if entry.type == "output_text" and results["thought"] is None:
                results["thought"] = entry.text
            
            elif entry.type == 'function_call' and entry.name == 'think_reflect' and results['thought'] is None:
                args = json.loads(entry.arguments)
                results['thought'] = args.get('note', '')

            elif entry.type == "function_call" and results["tool"] is None:
                tool_call = entry
                results["tool"] = tool_call.name

                try:
                    if tool_call.arguments:
                        args = json.loads(tool_call.arguments)
                        results["args"] = args if isinstance(args, dict) else {}
                    else:
                        results["args"] = {}
                except Exception as e:
                    print(f"[Warning] Failed to parse tool arguments: {e}")
                    results["args"] = {}
            
            # Optional: break if both are found
            if results["thought"] and results["tool"]:
                break

        return results
    
    def judge_step(self, step: str, trace: List[Dict]):
        """
        Use the LLM to decide if a step is complete based on reasoning trace and last observation.
        Returns: True if complete, False otherwise.
        """
        history_str = "\n".join([
            f"Thought: {t['thought']}\nAction: {t['action']}\nObservation: {t['observation']}"
            for t in trace[-3:]
        ])
        prompt = (
            f"You are an enterprise data scientist following a step by step plan to answer a business question.\n\n"
            f"Current step:\n'{step}'\n\n"
            f"Recent trace:\n{history_str}\n\n"
            f"Question: Has the step been completed successfully?\n"
            f"Answer 'yes' or 'no' and briefly justify."
        ) 

        response = self._call_llm(prompt)
        print(response)

        return "yes" in response.lower()[:10]



