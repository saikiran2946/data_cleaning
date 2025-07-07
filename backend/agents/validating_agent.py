from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent
import json

class ValidatingAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = f"""
You are an advanced, context-aware Data Quality Validator and Fixer. Analyze the dataset profile below and list all remaining data quality issues found **after all other cleaning agents have run**.

You must:
- Fully analyze and leverage the provided data dictionary and dataset profile. Do not rely on hardcoded rules or assumptions.
- Make all validation decisions dynamically and contextually, adapting your logic to the specific dataset and its data dictionary.
- For each column, consider its description, type, allowed values, and any constraints from the data dictionary.
- For each issue, suggest the best fix and provide a single line of Python code to fix it (for a pandas DataFrame named 'df').
- Only suggest fixes for issues that remain after all other agents have run.

For each issue, you must include:
- "column": The exact column name (or "__table__" for table-level issues like duplicates).
- "type": The type of issue (e.g., missing_value, outlier, type_mismatch, duplicate, etc.).
- "suggested_action": The best action to fix this issue (e.g., fillna_mean, drop_duplicates, clip_to_bounds, etc.).
- "reason": A clear, context-based explanation of the issue and why this fix is appropriate.
- "code": A single line of Python code to fix the issue (for a pandas DataFrame named 'df').

COLUMNS IN DATASET:
{list(self.df.columns)}

DATA DICTIONARY CONTEXT:
{self.get_data_dictionary_context()}

DATASET PROFILE:
{input}

Output a JSON object with a key "columns" (list of issues/fixes). Each item must have:
- "column": The column name (or "__table__" for table-level issues)
- "type": The type of issue
- "suggested_action": The best action to fix this issue
- "reason": Description of the issue and why this fix is appropriate
- "code": Python code to fix the issue (for a pandas DataFrame named 'df')

Return only valid JSON inside a single markdown code block.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        # Use a detailed profile for validation
        profile = []
        for col in self.df.columns:
            missing_count = int(self.df[col].isnull().sum())
            dtype = str(self.df[col].dtype)
            try:
                unique_count = self.df[col].astype(str).nunique(dropna=True)
                sample_values = self.df[col].dropna().astype(str).unique()[:10].tolist()
            except Exception as e:
                unique_count = f"Error: {e}"
                sample_values = [f"Error: {e}"]
            profile.append({
                "name": col,
                "dtype": dtype,
                "missing_count": missing_count,
                "unique_count": unique_count,
                "sample_values": sample_values,
            })
        # Table-level info
        table_profile = {
            "total_rows": len(self.df),
            "duplicate_rows": int(self.df.astype(str).duplicated().sum()),
            "columns": list(self.df.columns),
        }
        return {"columns": profile, "table": table_profile}

    def _parse_llm_response(self, response: str, profile: dict):
        import json
        import re
        try:
            cleaned_response = response.strip()
            match = re.search(r"```json\n(.*?)\n```", cleaned_response, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = cleaned_response
            if not json_str or json_str == cleaned_response:
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return []
            data = json.loads(json_str)
            actions = data.get("columns", [])
            # Ensure all required fields are present
            for action in actions:
                action.setdefault("column", "unknown")
                action.setdefault("type", "unknown")
                action.setdefault("suggested_action", "skip")
                action.setdefault("reason", "No reason provided.")
                action.setdefault("code", "# No code provided.")
            return actions
        except Exception as e:
            return []

    def generate_code_from_choice(self, column_name, choice: dict) -> str:
        # Return the code provided by the LLM for this fix
        return choice.get("code", "# No code provided.")