import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import snowflake.connector
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from llm.prompts import dbschema_str
import agent.tool_utils as tool_utils
import joblib
import contextlib
import traceback
import io
import re

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def convert_text_to_sql(text: str):
    # Connect to Snowflake
    try:
        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USERNAME'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA'),
        )
        cur = conn.cursor()
    except Exception as e:
        print("Failed to connect to Snowflake:", e)
        return None

    # Prompt setup
    system_prompt = (
        f"You are an expert data engineer who transforms a natural language query into a SQL query for Snowflake.\n\n"
        f"Here is the database schema:\n{dbschema_str}\n\n"
        f"Respond with just the generated SQL code, and nothing else (no backticks, no explanations, no comments)."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        )
        sql = response.choices[0].message.content.strip()
        print("Generated SQL:\n", sql)
    except Exception as e:
        print("Failed to get response from OpenAI:", e)
        return None

    try:
        cur.execute(sql)
        columns = [col[0] for col in cur.description] if cur.description else []
        rows = cur.fetchall()

        # Detect scalar or table result
        if len(rows) == 1 and len(columns) == 1:
            print("Scalar result:", rows[0][0])
            return rows[0][0]
        else:
            df = pd.DataFrame(rows, columns=columns)
            print("Tabular result (top rows):\n", df.head())
            return df
    except Exception as e:
        print("Failed to execute SQL:", e)
    finally:
        cur.close()
        conn.close()


# Main prediction function
def predict_clv_for_users(user_ids: pd.DataFrame, model_path=None):
    if model_path is None:
        # Resolve project root dynamically (2 levels up from this file)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(project_root, "models", "future_clv_model.joblib")

    conn = tool_utils.connect_to_snowflake()
    try:
        user_data = tool_utils.fetch_user_data(user_ids, conn)
        features = tool_utils.generate_clv_features(user_data)


        if features.empty:
            print("No valid feature data for given users.")
            return {}

        model = joblib.load(model_path)
        X = features[["recency", "frequency", "monetary", "avg_rating"]]
        preds = model.predict(X)

        return dict(zip(features["CUSTOMER_UNIQUE_ID"], preds))
    finally:
        conn.close()


# Main prediction function
def predict_churn_for_users(user_ids: pd.DataFrame, model_path=None):
    if model_path is None:
        # Resolve project root dynamically (2 levels up from this file)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(project_root, "models", "churn_model.joblib")

    conn = tool_utils.connect_to_snowflake()
    try:
        user_data = tool_utils.fetch_user_data(user_ids, conn)
        features = tool_utils.generate_churn_features(user_data)

        if features.empty:
            print("No valid feature data for given users.")
            return {}

        model = joblib.load(model_path)
        X = features[["recency", "frequency", "monetary", "avg_rating", "avg_shipping_delay"]]
        preds = model.predict(X)

        return dict(zip(features["CUSTOMER_UNIQUE_ID"], preds))
    finally:
        conn.close()


def write_python_code(prompt: str, params = None):
    """
    Generates and executes a Python function from a text prompt. Mostly to apply ad-hoc data transformations & analysis
    
    Args:
        prompt (str): Natural language prompt describing the desired function.
        params: (Optional) data that will act as input to the function 
    
    Returns:
        Dict containing:
        - 'code': str, generated Python code
        - 'output': str, stdout from execution
        - 'error': str, traceback if an error occurred
    """
    # Step 1: Generate code using LLM
    gen_prompt = f"Write a single, complete Python function for this task:\n{prompt}\nOnly output valid code. Do NOT include markdown/code fences, explanation, or backticks."
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": gen_prompt},
            ],
            temperature=0.1,
        )
        code = response.choices[0].message.content.strip()
    except Exception as e:
        return None

    print("Generated code:\n", code)
    # Step 2: Execute code and capture output
    exec_env = {}
    output_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            exec(code, exec_env)

            # If parameters are passed, find and call the first function
            if params is not None:
                # Try to find the first defined function name
                match = re.search(r'def\s+(\w+)\s*\(', code)
                if match:
                    func_name = match.group(1)
                    func = exec_env.get(func_name)
                    if callable(func):
                        # If params is a dictionary, unpack as keyword args
                        if isinstance(params, dict):
                            result = func(**params)
                        else:
                            result = func(params)
                        print(result)

        print("Execution output:\n", output_buffer.getvalue().strip())
        return output_buffer.getvalue().strip(),
     

    except Exception:
        return None


tool_specs = [
    {
        "type": "function",
        "name": "convert_text_to_sql",
        "description": "Transforms a natural language query into a SQL query for Snowflake and executes the query.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": [
                "text",
                "output_var"
            ],
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Natural language query to be converted into SQL"
                },
                "output_var": {
                    "type": "string",
                    "description": "Name of the variable to store the SQL query result"
                }
            },
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "predict_clv_for_users",
        "description": "Predicts customer lifetime value (CLV) for given user IDs.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": [
                "user_ids_var",
                "output_var"
            ],
            "properties": {
                "user_ids_var": {
                    "type": "string",
                    # "items": {
                    #     "type": "string"
                    # },
                    "description": "Name of variable storing a pandas Series of user IDs for which to predict CLV"
                },
                "output_var" : {
                    "type": "string",
                    "description": "Name of the variable to store the predictions"
                }
            },
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "predict_churn_for_users",
        "description": "Predicts churn likeliehood for given user IDs.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": [
                "user_ids_var",
                "output_var"
            ],
            "properties": {
                "user_ids_var": {
                    "type": "string",
                    "description": "Name of variable storing the list of user IDs for which to predict churn"
                },
                "output_var": {
                    "type": "string",
                    "description": "Name of the variable to store the predictions"
                }
            },
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "write_python_code",
        "description": "Generates and executes a Python function, mostly for data analysis or wrangling. ",
        "parameters": {
            "type": "object",
            "required": ["prompt", "params_var", "output_var"],
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Natural language prompt describing the desired code."
                },
                "params_var": {
                    "type": "string",
                    "description": "Name of a single scratchpad variable or a comma-separated string of scratchpad variables. Variables will be unpacked as keyword arguments if possible."
                },
                "output_var": {
                    "type": "string",
                    "description": "Name of the variable to store the function output"
                }
            },
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "think_reflect",
        "description": "Use this when you want to reflect, reason, or summarize your current thinking without taking any action.",
        "parameters": {
            "type": "object",
            "required": ["note"],
            "properties": {
                "note": {
                    "type": "string",
                    "description": "Natural language explanation or description of your thought process (e.g., why no action was taken)."
                }
            },
            "additionalProperties": False
        }
    }
]

tool_mapper = {
    "predict_churn_for_users": predict_churn_for_users,
    "predict_clv_for_users": predict_clv_for_users,
    "convert_text_to_sql": convert_text_to_sql,
    "write_python_code": write_python_code
}