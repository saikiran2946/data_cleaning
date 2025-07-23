import numpy as np
from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent

class OutlierAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = f"""
ROLE:
You are an advanced, context-aware Data Cleaning Agent specializing in outlier detection and treatment for tabular datasets.

You must always:
- Analyze the dataset, data dictionary, and column names before making any decision.
- Infer the intended data type and business meaning for each column from the data, business context, and data dictionary, not just the pandas dtype.
- Justify every outlier action and code with reference to the dataset's context, business meaning, and data dictionary.

ACTION:
For every column, you must:
- Analyze the column's data type, business meaning, unique values, and any constraints from the data dictionary.
- Critically evaluate the necessity and impact of outlier treatment for each column. Only recommend outlier treatment if it is clearly beneficial for data quality, analysis, or business outcomes. If the benefit is marginal or uncertain, default to "skip" and explain why.
- Provide a context-based reason for your choice, referencing the dataset, data dictionary, and column meaning.
- If you recommend a specific method (e.g., clipping, winsorizing, removing, flagging), ensure the code is type-safe and will not cause errors for non-numeric or mixed-type columns.

CONSTRAINTS:
- Never drop any column, regardless of outlier percentage or context.
- Use the data dictionary and dataset profile for all decisions; do not rely on hardcoded rules.
- Make decisions dynamically and contextually for each column, always considering the intended data type and business meaning.
- The "suggested_action" must be the best fit for each column; do not use the same action for all columns.
- For categorical or text columns, do not apply numeric outlier methods; explain why outlier treatment is not applicable.
- For date/datetime columns, only apply outlier logic if it makes sense in the business context, and ensure the code is robust to string or datetime types.
- For numeric columns, ensure the code is robust to mixed types (e.g., strings, NaNs) and does not cause type errors.
- Only recommend outlier treatment if it will clearly improve data quality, analysis, or business outcomes. If the benefit is marginal or could harm interpretability or downstream use, default to "skip" and justify your reasoning.
- Your output should show a variety of actions, reflecting the true context and needs of each column.
- Every decision must include a valid, context-based reason that references the dataset and data dictionary, and explicitly states the impact of the action on data quality, analysis, or business outcomes.
- Choose from: "skip", "clip_to_bounds", "winsorize", "remove_outliers", "flag_outliers". Never use "drop_column".

OUTPUT:
- Return your response as a single JSON object inside a markdown code block.
- The JSON must have a key "columns", whose value is a list of objects, one per column.
- Each object must contain:
  - "name": The exact column name (must match one from the dataset).
  - "suggested_action": The best outlier treatment (from the allowed actions).
  - "constant_value": If needed, specify the value; otherwise, leave empty.
  - "reason": A clear, context-based explanation for your decision, referencing the dataset and data dictionary, and explicitly stating the impact of the action.
  - "code": Python code to perform the outlier treatment, robust to the column's data type and context.

Example Output Format (as plain text, not a code block):

Output:
columns: [
  name: Age, suggested_action: clip_to_bounds, constant_value: , reason: Age is numeric and typically ranges from 0 to 120; clipping to these bounds removes data entry errors and improves data quality for analysis. code: df['Age'] = pd.to_numeric(df['Age'], errors='coerce'); df['Age'] = df['Age'].clip(lower=0, upper=120)
  name: Salary, suggested_action: skip, constant_value: , reason: Salary has a wide natural range and outlier treatment may remove valid high earners; skipping preserves business insight and avoids harming downstream analysis. code: # No action chosen.
]

- Do not include any text or explanation outside the code block.

COLUMNS IN DATASET:
{list(self.df.columns)}

DATA DICTIONARY CONTEXT:
{self.get_data_dictionary_context()}

DATASET PROFILE:
{input}
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        profile = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            stats = self.df[col].describe()
            q1 = stats["25%"]
            q3 = stats["75%"]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = int(outlier_mask.sum())
            profile.append({
                "name": col,
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "min": float(stats["min"]),
                "25%": float(stats["25%"]),
                "50%": float(stats["50%"]),
                "75%": float(stats["75%"]),
                "max": float(stats["max"]),
                "outlier_count": outlier_count
            })
        return profile 

    def _parse_llm_response(self, response: str, profile: list):
        import json
        try:
            # Clean the response first
            cleaned_response = response.strip()
            # Try to extract JSON from markdown code blocks
            import re
            match = re.search(r"```json\n(.*?)\n```", cleaned_response, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = cleaned_response
            # If no JSON found in code blocks, try to find JSON in the response
            if not json_str or json_str == cleaned_response:
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # If no JSON found, use outlier_count to determine action
                    actions = []
                    for col in profile:
                        outlier_count = col.get("outlier_count", 0)
                        if outlier_count > 0:
                            actions.append({
                                "name": col.get("name"),
                                "suggested_action": "clip_to_bounds",
                                "reason": f"{outlier_count} outliers detected; recommend handling outliers (e.g., clip_to_bounds).",
                                "outlier_count": outlier_count
                            })
                        else:
                            actions.append({
                                "name": col.get("name"),
                                "suggested_action": "skip",
                                "reason": "No outliers detected; no action needed for this dataset.",
                                "outlier_count": 0
                            })
                    return actions
            # Parse the JSON
            data = json.loads(json_str)
            actions = data.get("columns", [])
            # If actions is empty, use outlier_count to determine action
            if not actions:
                actions = []
                for col in profile:
                    outlier_count = col.get("outlier_count", 0)
                    if outlier_count > 0:
                        actions.append({
                            "name": col.get("name"),
                            "suggested_action": "clip_to_bounds",
                            "reason": f"{outlier_count} outliers detected; recommend handling outliers (e.g., clip_to_bounds).",
                            "outlier_count": outlier_count
                        })
                    else:
                        actions.append({
                            "name": col.get("name"),
                            "suggested_action": "skip",
                            "reason": "No outliers detected; no action needed for this dataset.",
                            "outlier_count": 0
                        })
                return actions
            # Ensure outlier_count is included in each action
            for i, col_action in enumerate(actions):
                if i < len(profile):
                    col_action.setdefault("name", profile[i].get("name"))
                    col_action.setdefault("outlier_count", profile[i].get("outlier_count", 0))
                col_action.setdefault("reason", "No reason provided by LLM.")
                col_action.setdefault("suggested_action", "skip")
            return actions
        except Exception as e:
            # On error, use outlier_count to determine action
            actions = []
            for col in profile:
                outlier_count = col.get("outlier_count", 0)
                if outlier_count > 0:
                    actions.append({
                        "name": col.get("name"),
                        "suggested_action": "clip_to_bounds",
                        "reason": f"{outlier_count} outliers detected; recommend handling outliers (e.g., clip_to_bounds).",
                        "outlier_count": outlier_count
                    })
                else:
                    actions.append({
                        "name": col.get("name"),
                        "suggested_action": "skip",
                        "reason": "No outliers detected; no action needed for this dataset.",
                        "outlier_count": 0
                    })
            return actions 