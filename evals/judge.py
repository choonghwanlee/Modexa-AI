from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI()


class PlanScore(BaseModel):
    conciceness: int
    feasibility: int
    effectiveness: int
    rationale: str

class ResponseScore(BaseModel):
    helpfulness: int
    rationale: str


def eval_plan(question, plan):
    prompt = f"""
    You are a critical-thinking research assistant evaluating a proposed plan to answer a data question.

    **Question:**
    {question}

    **Plan Steps:**
    {chr(10).join(f"{i+1}. {step}" for i, step in enumerate(plan))}

    Evaluate the plan on a scale from 1 to 5 for:
    1. **Conciseness** – Are there unnecessary or redundant steps?
    2. **Feasibility** – Are the tools/data/models used realistic and available?
    3. **Effectiveness** – Will these steps retrieve the right information to answer the question?

    Include a brief rationale for each score

    """
    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text_format=PlanScore,
    )
    evals = response.output_parsed
    return evals


def eval_response(question, response):
    prompt = f"""
    You are a critical-thinking research assistant evaluating an AI agent's response to a data question.

    **Question:**
    {question}

    **Response:**
    {response}

    Evaluate the plan on a scale from 1 to 5 for:
    1. **Helpfulness** – Does the response directly and completely answer the user's question?

    Include a brief rationale for the score
    """
    response = client.responses.parse(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text_format=ResponseScore,
    )
    evals = response.output_parsed
    return evals

