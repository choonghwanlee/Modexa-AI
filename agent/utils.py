import pandas as pd
import ast


def summarize_value(value, preview_items: int = 3) -> str:
    """
    Provides a smart, readable summary of a scratchpad value.
    Supports DataFrames, lists, dicts, and generic fallbacks.
    """

    if isinstance(value, pd.DataFrame):
        return summarize_dataframe(value, preview_rows=0)

    elif isinstance(value, list):
        summary = f"<list with {len(value)} items>"
        if value:
            sample_items = "\n".join([f"- {repr(item)}" for item in value[:preview_items]])
            summary += f"\nSample:\n{sample_items}"
        return summary

    elif isinstance(value, dict):
        summary = f"<dict with {len(value)} keys>"
        if value:
            sample_items = "\n".join(
                [f"- {k}: {repr(value[k])}" for k in list(value.keys())[:preview_items]]
            )
            summary += f"\nSample:\n{sample_items}"
        return summary

    elif isinstance(value, (str, int, float, bool)):
        return repr(value)

    return f"<{type(value).__name__}>: {str(value)[:200]}"

def summarize_dataframe(df: pd.DataFrame, preview_rows: int = 0) -> str:
    """
    Generate a human-readable summary of a pandas DataFrame.
    
    This function includes:
    - Shape (number of rows and columns)
    - Column names with their data types
    - An optional preview of the first N rows
    - A note about any missing values

    Args:
        df (pd.DataFrame): The DataFrame to summarize.
        preview_rows (int, optional): Number of rows to preview (as markdown table). Defaults to 0 (no preview).

    Returns:
        str: A multiline string summary of the DataFrame, suitable for LLM prompts or logs.
    """
    # Get basic shape and column metadata

    shape_info = f"{df.shape[0]} rows × {df.shape[1]} cols"
    column_info = ", ".join([f"{col}({dtype})" for col, dtype in df.dtypes.items()])
    column_info = f"Columns: {column_info}"

    summary = f"<DataFrame: {shape_info}>\n{column_info}"

    # Optional: add preview of values
    if preview_rows > 0:
        preview = df.head(preview_rows).to_markdown(index=False)
        summary += f"\nPreview:\n{preview}"

    # Optional: include missing value stats
    if df.isnull().values.any():
        na_cols = df.columns[df.isnull().any()].tolist()
        summary += f"\nNote: Missing values in {', '.join(na_cols)}"

    return summary


def resolve_args_from_scratchpad(args: dict, scratchpad: dict) -> dict:
    """
    Replaces *_var entries in args with actual objects from the scratchpad.
    
    Example:
        args = {
            "input_df_var": "df_q4_churn",
            "model_name": "churn_predictor"
        }
        scratchpad = {
            "df_q4_churn": <pandas.DataFrame>
        }

        Returns:
            {
                "input_df": <pandas.DataFrame>,
                "model_name": "churn_predictor"
            }
    """
    resolved = {}
    for key, value in args.items():
        if key.endswith("_var"):
            target_key = key[:-4]

            # Handle string of comma-separated variable names
            if isinstance(value, str) and "," in value:
                keys = [v.strip() for v in value.split(",")]
                actual_dict = {}
                for k in keys:
                    if k not in scratchpad:
                        raise ValueError(f"Scratchpad variable '{k}' not found for key '{key}'")
                    actual_dict[k] = scratchpad[k]
                resolved[target_key] = actual_dict
                continue

            # Handle regular single string reference
            actual_value = scratchpad.get(value)
            if actual_value is None:
                raise ValueError(f"Scratchpad variable '{value}' not found for key '{key}'")
            resolved[target_key] = actual_value

        else:
            resolved[key] = value

    return resolved
