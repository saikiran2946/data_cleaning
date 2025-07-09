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
                uniques_full = self.df[col].dropna().astype(str).unique().tolist()
                uniques = uniques_full[:20]
                print(f"[DEBUG] Unique values for {col}: {uniques_full}")
                sample_lines.append(f"- {col}: {uniques}")
                if len(uniques_full) > 100:
                    sample_lines.append(f"  [WARNING: Only first 100 of {len(uniques_full)} unique values shown.]")
            except Exception as e:
                sample_lines.append(f"- {col}: [Could not extract unique values: {e}]")
        sample_block = "\n".join(sample_lines)
        prompt = f"""
You are an expert Data Engineer responsible for standardizing values in a dataset for robust, automated data cleaning.

**Available columns in dataset:** {list(self.df.columns)}

**Sample unique values for each column:**
{sample_block}

**MANDATORY INSTRUCTIONS:**
- For EVERY column, you MUST analyze ALL unique values (not just the sample above) and suggest mappings for ANY value that is not canonical, standard, or valid for the column's type and business context.
- For every column, you MUST identify and expand all abbreviations, acronyms, and short forms to their full, canonical values (e.g., 'NY' → 'New York', 'LA' → 'Los Angeles', 'M' → 'Male', etc.).
- Only suggest mappings for values that are actually inconsistent, ambiguous, or non-canonical in the context of the dataset.
- Do NOT suggest unnecessary mappings for values that are already correct, valid, and meaningful. Carefully check all unique values and only provide useful, meaningful standardization suggestions that improve data quality and consistency.
- Do not suggest mappings for values that are already correct, valid, and meaningful in the dataset context.
- You are FORBIDDEN from skipping a column unless EVERY value is already canonical and valid. If you skip, you MUST list ALL unique values for that column as proof and explain why each is canonical.
- If you find even ONE non-canonical, non-standard, placeholder, or ambiguous value, you MUST suggest a mapping for it. If in doubt, err on the side of suggesting a mapping.
- If you skip standardization for a column that contains any non-canonical value, you will be penalized.
- 'No value standardization needed' is ONLY allowed if all values are already canonical, and you MUST list all unique values to prove it.
- Be exhaustive and conservative: always check for abbreviations, synonyms, typos, case variations, placeholders, and ambiguous values.
- For categorical/text columns: Map all abbreviations, case variations, typos, synonyms, and partial matches to a single, contextually appropriate canonical value.
- For numeric columns: Map all non-numeric, placeholder, or corrupted values to NaN or a valid missing value indicator.
- For date/datetime columns: Map all invalid, placeholder, or ambiguous values to NaT or a valid missing value indicator.
- For boolean columns: Map all variations (e.g., 'yes', 'Y', '1', 'true', 'True') to True, and ('no', 'N', '0', 'false', 'False') to False.
- For all columns: Be exhaustive—do not leave any value unmapped if it is a variation, abbreviation, typo, or placeholder.
- If a value is already valid and canonical, map it to itself (do not include it in the output mapping).
- After mapping, there should be no abbreviations, case variations, placeholders, or invalid values left—only canonical or valid values for the column type.
- Only include mappings for values that need to be changed. If a value is already canonical, do not include it in the output mapping.
- You MUST check for abbreviations, synonyms, typos, and case variations in ALL unique values, not just the sample above. If the column has more than 100 unique values, assume there are additional variations to standardize and do not skip standardization if any non-canonical value is present.
- You MUST NOT miss or skip any value that is an abbreviation, typo, synonym, case variation, placeholder, or ambiguous.
- If there is any doubt about a value’s canonical status, you MUST include it in the mappings and provide your best suggestion for standardization.
- If you are unsure, err on the side of suggesting a mapping.
- If you skip a value that should have been standardized, you will be penalized.
- Review all unique values for every column and ensure every non-canonical, inconsistent, or unclear value is mapped.
- If a value is already canonical, do not include it in the output mapping.

**Contextual Explanation:**
- For each mapping, provide a clear, context-based explanation for why the mapping is necessary and how it improves data quality, consistency, or downstream analysis.
- Use the column's data type, business meaning, and sample values to justify your decisions.

**Examples:**
- For cities: "LA", "la", "L.A.", "Los Angeles", "los angeles" → "Los Angeles"; "NYC", "nyc", "New York", "new york", "NY" → "New York"
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
        mapping_dict = {}
        for m in mappings:
            from_val = str(m["from"]).strip()
            to_val = m["to"]
            if from_val != to_val:
                # Convert JSON null/None/empty string/nan to Python None
                if to_val in [None, "null", "None", "", "nan"]:
                    mapping_dict[from_val] = None
                else:
                    mapping_dict[from_val] = to_val
        if not mapping_dict:
            return ""
        code_lines = [
            f"df[{repr(column_name)}] = df[{repr(column_name)}].astype(str).str.strip().replace({mapping_dict})"
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
            print(f"[DEBUG] LLM response: {response}")
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            actions = []
            # --- Handle dict-of-dicts format (column name → mapping dict) ---
            if isinstance(data, dict) and not data.get("columns"):
                for col, mapping_dict in data.items():
                    if isinstance(mapping_dict, dict):
                        mappings = [
                            {"from": k, "to": v} for k, v in mapping_dict.items() if k != v
                        ]
                        actions.append({
                            "name": col,
                            "mappings": mappings,
                            "suggested_action": "standardize_values" if mappings else "skip",
                            "reason": f"Standardization needed for {col}" if mappings else f"No standardization needed for {col}"
                        })
            # --- Handle list-of-actions format ---
            if "columns" in data:
                actions.extend(data.get("columns", []))
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