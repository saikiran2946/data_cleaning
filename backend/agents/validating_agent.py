from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent
import json

class ValidatingAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = f"""
You are an advanced, context-aware Data Quality Validator. Analyze the dataset profile below and list all data quality issues found.

You must:
- Fully analyze and leverage the provided data dictionary and dataset profile. Do not rely on hardcoded rules or assumptions.
- Make all validation decisions dynamically and contextually, adapting your logic to the specific dataset and its data dictionary.
- Use principles of robust, explainable, and context-driven data validation.
- For each column, consider its description, type, allowed values, and any constraints from the data dictionary.
- Suggest only issues that are truly relevant and important, with clear reasoning based on the actual context.

For each issue, you must include:
- "column": The exact column name (or "__table__" for table-level issues like duplicates).
- "type": The type of issue (e.g., missing_value, outlier, type_mismatch, duplicate, etc.).
- "suggested_agent": The best agent to fix this issue (choose from: "Missing Value", "Outliers", "Data Types", "Duplicates", "Normalization", "Value Standardization", "Feature Generation").
- "description": A clear, context-based description of the issue.
- "severity": high/medium/low.

COLUMNS IN DATASET:
{list(self.df.columns)}

DATA DICTIONARY CONTEXT:
{self.get_data_dictionary_context()}

DATASET PROFILE:
{input}

Options for validation (select the best based on your reasoning): ["status": "completed" or "issues found", with a list of issues]

Output a JSON object with a key "status" and a list of "issues". Each issue must have:
- "column": The column name (or "__table__" for table-level issues)
- "type": The type of issue
- "suggested_agent": The best agent to fix this issue
- "description": Description of the issue
- "severity": high/medium/low

Return only valid JSON inside a single markdown code block.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        return [{
            "missing_values": self.df.isnull().sum().sum(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "data_types": self.df.dtypes.to_string(),
        }]