from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent
import json

class GeneralIssueAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = f"""
You are an advanced, context-aware Data Cleaner. You are given a description of a data quality issue. Suggest a fix and generate Python code to resolve it on a pandas DataFrame named 'df'.

You must:
- Fully analyze and leverage the provided data dictionary and dataset profile. Do not rely on hardcoded rules or assumptions.
- Make all decisions dynamically and contextually, adapting your logic to the specific dataset and its data dictionary.
- Use principles of robust, explainable, and context-driven data cleaning.
- For each column, consider its description, type, allowed values, and any constraints from the data dictionary.
- Suggest only fixes that are truly relevant and important, with clear reasoning based on the actual context.

COLUMNS IN DATASET:
{list(self.df.columns)}

DATA DICTIONARY CONTEXT:
{self.get_data_dictionary_context()}

ISSUE DESCRIPTION:
{input}

Output a JSON object with a key "fix" and a key "code". Use only column names from the COLUMNS IN DATASET list. Provide clear, context-based reasoning for your decision.

Return only valid JSON inside a single markdown code block.
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def profile_columns(self):
        return [{"issue": "User-described data quality issue"}]

    def _parse_llm_response(self, response: str, profile: list):
        """
        The response for this agent is a single JSON object, not a list.
        We wrap it in a list to conform to the base agent's expectation.
        """
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            return [data] 
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing LLM response for {self.__class__.__name__}: {e}")
            return [{"fix": "Parsing Error", "code": "# Could not parse LLM response."}]