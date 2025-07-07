import json
import re
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent

class ValueStandardizationAgent(BaseAgent):
    """
    Agent for standardizing values in a DataFrame, including categorical and numeric columns.
    """
    # =========================
    # Prompt Template
    # =========================
    def _get_prompt_template(self) -> ChatPromptTemplate:
        # Gather unique values for each column (sampled from the data)
        sample_lines = []
        for col in self.df.columns:
            try:
                uniques = self.df[col].dropna().astype(str).unique()[:10].tolist()
                sample_lines.append(f"- {col}: {uniques}")
            except Exception as e:
                sample_lines.append(f"- {col}: [Could not extract unique values: {e}]")
        sample_block = "\n".join(sample_lines)
        prompt = f"""
You are an expert Data Engineer responsible for standardizing values in a dataset for robust, automated data cleaning.

**Available columns in dataset:** {list(self.df.columns)}

**Sample unique values for each column:**
{sample_block}

For each column, you MUST:
- Only process columns that are present in the dataset (see list above).
- Analyze the column's data type, unique values (see sample above), and business context.
- Map all non-standard, inconsistent, placeholder, or ambiguous values (e.g., 'Unknown', 'N/A', '-', '', 'null', 'none', 'missing', etc.) to either a canonical value or to NaN/empty string as appropriate for the column type.
- For categorical/text columns: Map all abbreviations, case variations, typos, synonyms, and partial matches to a single, contextually appropriate canonical value.
- For numeric columns: Map all non-numeric, placeholder, or corrupted values to NaN or a valid missing value indicator.
- For date/datetime columns: Map all invalid, placeholder, or ambiguous values to NaT or a valid missing value indicator.
- For boolean columns: Map all variations (e.g., 'yes', 'Y', '1', 'true', 'True') to True, and ('no', 'N', '0', 'false', 'False') to False.
- For all columns: Be exhaustive—do not leave any value unmapped if it is a variation, abbreviation, typo, or placeholder.
- If a value is already valid and canonical, map it to itself.
- After mapping, there should be no abbreviations, case variations, placeholders, or invalid values left—only canonical or valid values for the column type.
- **Only include mappings for values that need to be changed. If a value is already canonical, do not include it in the output mapping.**

**Contextual Explanation:**
- For each mapping, provide a clear, context-based explanation for why the mapping is necessary and how it improves data quality, consistency, or downstream analysis.
- Use the column's data type, business meaning, and sample values to justify your decisions.

**Examples:**
- For cities: "LA", "la", "L.A.", "Los Angeles", "los angeles" → "Los Angeles"; "NYC", "nyc", "New York", "new york" → "New York"
- For gender: "M", "m", "Male", "male" → "Male"; "F", "f", "Female", "female" → "Female"
- For salary: 'Unknown', 'N/A', '-', '' → NaN
- For boolean: 'yes', 'Y', '1', 'true', 'True' → True; 'no', 'N', '0', 'false', 'False' → False

Return your response inside a single markdown code block as a JSON object.

- Return only valid JSON.
- Use only column names from the available columns list above.
- Provide exhaustive mappings and explanations that ensure only canonical or valid values remain after mapping.
- If you skip a column, explain exactly why no standardization is needed for that column, referencing the actual unique values.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    # =========================
    # Profile Generation
    # =========================
    def profile_columns(self):
        profile = []
        for col in self.df.columns:
            try:
                unique_values = list(self.df[col].dropna().astype(str).unique())
            except Exception as e:
                unique_values = [f"Could not extract unique values: {e}"]
            if len(unique_values) > 0:
                profile.append({
                    "name": col,
                    "unique_values": unique_values,
                    "num_unique": len(unique_values),
                    "total_count": len(self.df[col]),
                    "null_count": self.df[col].isnull().sum()
                })
        return {
            "columns_in_dataset": list(self.df.columns),
            "profile": profile
        }

    # =========================
    # Code Generation
    # =========================
    def generate_code_from_choice(self, column_name, choice):
        if column_name not in self.df.columns:
            return ""
        mappings = choice.get("mappings", [])
        if not mappings:
            return ""
        mapping_dict = {str(m["from"]).strip(): m["to"] for m in mappings if str(m["from"]).strip() != m["to"]}
        if not mapping_dict:
            return ""
        code_lines = [
            f"df[{repr(column_name)}] = df[{repr(column_name)}].astype(str).str.strip().replace({json.dumps(mapping_dict)})"
        ]
        return "\n".join(code_lines)

    # =========================
    # LLM Response Parsing
    # =========================
    def _extract_json(self, response: str) -> str:
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def _parse_llm_response(self, response: str, profile: list):
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            actions = data.get("columns", [])
            # Handle new format: 'mappings' is a dict of dicts
            if not actions and "mappings" in data:
                for col, mapping_dict in data["mappings"].items():
                    mappings = [
                        {"from": k, "to": v} for k, v in mapping_dict.items()
                    ] if mapping_dict else []
                    actions.append({
                        "name": col,
                        "suggested_action": "standardize_values" if mappings else "skip",
                        "reason": data.get("explanations", {}).get(col, "No reason provided."),
                        "mappings": mappings
                    })
            valid_actions = []
            for action in actions:
                if "name" not in action:
                    action["name"] = "unknown_column"
                if "suggested_action" not in action:
                    action["suggested_action"] = "skip"
                if "reason" not in action:
                    action["reason"] = "No reason provided."
                if "mappings" not in action:
                    action["mappings"] = []
                if action["name"] in self.df.columns:
                    valid_actions.append(action)
                else:
                    print(f"[DEBUG] Skipping action for column '{action['name']}' (not in DataFrame columns). Reason: {action.get('reason', 'No reason')}")
                if action["suggested_action"] == "skip":
                    print(f"[DEBUG] Skipping column '{action['name']}' - Reason: {action['reason']}")
                else:
                    print(f"[DEBUG] Standardization action for column '{action['name']}': {len(action['mappings'])} mappings. Reason: {action['reason']}")
            return valid_actions
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._create_error_response(profile)

    def _create_error_response(self, profile: list) -> list:
        if not profile:
            return [{"name": "unknown", "suggested_action": "skip", "reason": "Agent failed due to an error."}]
        # Only include columns that are actually in the DataFrame
        real_columns = set(self.df.columns)
        return [
            {
                "name": col.get("name") if isinstance(col, dict) else col,
                "suggested_action": "skip",
                "reason": "Agent failed due to an error."
            }
            for col in profile
            if (col.get("name") if isinstance(col, dict) else col) in real_columns
        ] 