from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent
import json
import pandas as pd
import numpy as np
import re

class FeatureGenerationAgent(BaseAgent):
    def _extract_json(self, response: str) -> str:
        """Extracts a JSON string from a markdown code block."""
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = f"""
You are an advanced, context-aware Data Scientist. Your job is to suggest only the most important, analysis-relevant new features that can be created from the columns below.

You must:
- Fully analyze and leverage the provided data dictionary and dataset profile. Do not rely on hardcoded rules or assumptions.
- Make all feature generation decisions dynamically and contextually, adapting your logic to the specific dataset and its data dictionary.
- Use principles of robust, explainable, and context-driven feature engineering.
- For each column, consider its description, type, allowed values, and any constraints from the data dictionary.
- Suggest only features that are highly valuable for analysis or modeling, with clear reasoning based on the actual context.
- When writing formulas, always use only column names exactly as they appear in the dataset (COLUMNS IN DATASET).

**CRITICAL RULES:**
- Only suggest features that are highly valuable for analysis or modeling (e.g., total_amount = quantity * price, ratios, or domain-specific aggregations).
- DO NOT suggest trivial features like extracting year/month/day from dates, or one-hot encoding, or text length, unless they are absolutely critical for the dataset's context.
- For each feature, provide a clear name, the columns used, and a short reason why it is important.
- Suggest at most 3 new features, and only if they are truly useful.
- ALWAYS create meaningful feature names that describe what the feature represents.
- ALWAYS write formulas that use exact column names from the dataset.
- NEVER use column names that are not in the COLUMNS IN DATASET list.
- The "formula" must be a valid pandas expression, referencing columns from the COLUMNS IN DATASET list.
- Example: If creating a feature "price_per_sqft" from "price" and "sqft", the formula should be "df['price'] / df['sqft']". Always use the `df[]` syntax to refer to columns.
- Ensure the formula correctly handles potential division by zero or other errors by using `np.where` or other safe methods. For example: `np.where(df['sqft'] > 0, df['price'] / df['sqft'], 0)`.
- If a new feature requires multiple steps, combine them into a single-line formula.

COLUMNS IN DATASET:
{list(self.df.columns)}

DATA DICTIONARY CONTEXT:
{self.get_data_dictionary_context()}

DATASET PROFILE:
{input}

Options for feature generation (select the best based on your reasoning): ["skip", or provide a new feature with a "name", "formula" and "reason"]

Output a JSON object with a key "features". Each object in the list should contain:
- "name": A descriptive name for the new feature (e.g., "total_revenue", "customer_age", "price_per_unit")
- "formula": The Python expression to generate the feature using exact column names from the dataset
- "reason": A clear, context-based explanation for your decision

Return only valid JSON inside a single markdown code block.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        """Create a detailed profile of the dataset for feature generation analysis."""
        profile = []
        for col in self.df.columns:
            col_info = {
                "name": col,
                "dtype": str(self.df[col].dtype),
                "unique_count": self.df[col].nunique(),
                "null_count": self.df[col].isnull().sum(),
                "sample_values": self.df[col].dropna().head(5).tolist()
            }
            
            # Add numeric-specific info
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_info.update({
                    "min": float(self.df[col].min()) if not self.df[col].isnull().all() else None,
                    "max": float(self.df[col].max()) if not self.df[col].isnull().all() else None,
                    "mean": float(self.df[col].mean()) if not self.df[col].isnull().all() else None
                })
            
            # Add datetime-specific info
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_info["is_datetime"] = True
                col_info["date_range"] = {
                    "min": str(self.df[col].min()) if not self.df[col].isnull().all() else None,
                    "max": str(self.df[col].max()) if not self.df[col].isnull().all() else None
                }
            
            profile.append(col_info)
        
        return {
            "columns": profile,
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "numeric_columns": list(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": list(self.df.select_dtypes(include=['datetime64']).columns)
        }

    def _parse_llm_response(self, response: str, profile: list):
        """Override to parse a 'features' key instead of 'columns'."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            # Handle both "features" and "columns" keys for compatibility
            actions = data.get("features", data.get("columns", []))
            
            valid_actions = []
            for action in actions:
                # Ensure all required fields are present
                if "name" not in action or not action["name"]:
                    action["name"] = f"feature_{len(actions)}"
                if "formula" not in action:
                    action["formula"] = ""
                if "reason" not in action:
                    action["reason"] = "No reason provided."
                
                # Clean up the formula to ensure it's properly formatted
                formula = action["formula"]
                if formula and isinstance(formula, str):
                    # Remove any leading/trailing whitespace
                    formula = formula.strip()
                    # Ensure the formula doesn't already have column assignment
                    if not formula.startswith(f"df['{action['name']}']"):
                        action["formula"] = formula
                
                # Validate that the formula only uses existing column names
                if formula and isinstance(formula, str):
                    # Extract column names from the formula (simple regex for df['column'] or df["column"])
                    import re
                    column_matches = re.findall(r"df\[['\"]([^'\"]+)['\"]\]", formula)
                    invalid_columns = [col for col in column_matches if col not in self.df.columns]
                    
                    if invalid_columns:
                        continue
                    else:
                        pass
                
                valid_actions.append(action)
            
            return valid_actions
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing LLM response for {self.__class__.__name__}: {e}")
            print(f"Raw response: {response[:500]}...")
            return []

    def generate_code_from_choice(self, column_name, choice):
        """Generate Python code to create a new feature column."""
        feature_name = choice.get("name", column_name)
        formula = choice.get("formula", "")
        
        if not formula or not isinstance(formula, str):
            return ""
        
        formula = formula.strip()
        if not formula:
            return ""
        
        # Always create the new column by assignment, regardless of existence
        code = f"df['{feature_name}'] = {formula}"
        return code