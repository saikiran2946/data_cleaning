import pandas as pd
from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent
import re
import json

class DataTypeAgent(BaseAgent):
    """
    An agent that specializes in analyzing and correcting data types.
    """
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = f"""
You are an advanced, context-aware Data Cleaning Agent specializing in analyzing and correcting data types.

You must:
- Fully analyze and leverage the provided data dictionary and dataset profile. Do not rely on hardcoded rules or assumptions.
- Make all decisions dynamically and contextually, adapting your logic to the specific dataset and its data dictionary.
- Use principles of robust, explainable, and context-driven data cleaning.
- For each column, consider its description, type, allowed values, and any constraints from the data dictionary.
- Suggest the most appropriate data type for each column, with clear reasoning based on the actual context.

COLUMNS IN DATASET:
{list(self.df.columns)}

DATA DICTIONARY CONTEXT:
{self.get_data_dictionary_context()}

DATASET PROFILE:
{input}

Options for data type (select the best based on your reasoning): ["skip", "int64", "float64", "datetime64[ns]", "category", "string"]

Output a JSON object with a key "columns". Each object in the list should contain:
- "name": The exact column name (must match one from the COLUMNS IN DATASET list).
- "suggested_dtype": The best data type (choose from the options above).
- "reason": A clear, context-based explanation for your decision.

Return only valid JSON inside a single markdown code block.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        """Profiles columns for data type analysis."""
        profile = []
        for col in self.df.columns:
            unique_samples = self.df[col].dropna().unique()
            sample_size = min(5, len(unique_samples))
            samples = pd.Series(unique_samples).sample(sample_size).tolist() if sample_size > 0 else []

            profile.append({
                "name": col,
                "dtype": str(self.df[col].dtype),
                "sample_values": [str(s) for s in samples]
            })
        return profile

    def _parse_llm_response(self, response: str, profile: list):
        """Adds the original dtype to the parsed response for UI display."""
        actions = super()._parse_llm_response(response, profile)
        for col_profile, llm_action in zip(profile, actions):
            llm_action['dtype'] = col_profile['dtype']
        return actions 