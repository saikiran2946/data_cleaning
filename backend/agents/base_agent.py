import json
import re
import pandas as pd
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from utils.openai_client import llm

class BaseAgent:
    """
    A base class for all data cleaning agents, powered by LangChain.
    Each agent profiles data, suggests actions, and can generate code for those actions.
    Now enhanced with memory capabilities for context-aware decision making.
    """
    def __init__(self, dataframe: pd.DataFrame, data_dictionary=None, all_agents=None):
        self.df = dataframe
        self.data_dictionary = data_dictionary
        self.llm = llm # Use the centralized LLM
        self.agent_name = self.__class__.__name__.replace('Agent', '')
        
        # Each subclass must define its own prompt for suggesting actions
        action_prompt_template = self._get_prompt_template()
        
        # Create a simple chain for invoking the LLM, not a complex agent executor
        if action_prompt_template:
            self.action_chain: Runnable = action_prompt_template | self.llm
        else:
            self.action_chain = None

    def _get_prompt_template(self) -> ChatPromptTemplate:
        """
        Subclasses must implement this to provide their specific system prompt
        for **suggesting cleaning actions**.
        """
        raise NotImplementedError("Each agent must provide a prompt template for suggesting actions.")

    def _get_code_generation_prompt_template(self) -> ChatPromptTemplate:
        """
        Provides a standardized prompt for **generating Python code** based on a suggested action.
        """
        prompt = """
You are a Python code generation assistant. Your task is to write a single, executable line of pandas code to perform a specific data cleaning action on a DataFrame named 'df'.

- **DataFrame**: A pandas DataFrame named `df` is pre-defined and available.
- **Column**: You are working on the column named `{column_name}`.
- **Action**: The suggested action is: `{action_details}`
- **Reason**: The reason for this action is: `{reason}`

Based on this, generate a single line of python code. Do not add comments, explanations, or markdown. Output only the raw code.

Example for a 'fillna_median' action on column 'age':
df['age'].fillna(df['age'].median(), inplace=True)
"""
        return ChatPromptTemplate.from_messages([("system", prompt)])

    def _get_memory_enhanced_prompt_template(self) -> ChatPromptTemplate:
        """
        Provides a memory-enhanced prompt template that includes historical context.
        """
        base_prompt = self._get_prompt_template()
        if not base_prompt:
            return None
            
        # Get memory context
        memory_context = self._get_memory_context()
        
        # Extract the base prompt content correctly
        base_prompt_content = ""
        if hasattr(base_prompt, 'messages') and base_prompt.messages:
            # For ChatPromptTemplate, access the prompt content correctly
            if hasattr(base_prompt.messages[0], 'prompt'):
                base_prompt_content = base_prompt.messages[0].prompt.template
            elif hasattr(base_prompt.messages[0], 'content'):
                base_prompt_content = base_prompt.messages[0].content
            else:
                # Fallback: try to get the string representation
                base_prompt_content = str(base_prompt.messages[0])
        
        # Enhance the prompt with memory
        enhanced_prompt = f"""
{base_prompt_content}

MEMORY CONTEXT:
{memory_context}

Use this historical context to make better, more consistent decisions. Consider:
- Previous successful actions for similar data patterns
- User preferences and overrides
- Patterns in decision-making
- Lessons learned from past cleaning sessions
"""
        return ChatPromptTemplate.from_messages([("system", enhanced_prompt)])

    def _get_memory_context(self) -> str:
        """Get relevant memory context for this agent."""
        return "No memory context available."

    def profile_columns(self):
        """
        Subclasses must implement this to generate a profile of the dataframe
        relevant to their specific task.
        """
        raise NotImplementedError

    def generate_actions(self):
        """
        The main entry point for an agent to suggest cleaning actions.
        It profiles the data, invokes the LangChain chain, and parses the result.
        Now enhanced with memory context.
        """
        if not self.action_chain:
            return []

        profile = self.profile_columns()
        if not profile:
            return []
        
        profile_info = json.dumps(profile, indent=2, default=str)
        
        try:
            # Use only the default action_chain, as memory is not used
            chain_to_use = self.action_chain
            # Invoke the chain
            response = chain_to_use.invoke({"input": profile_info})
            output = response.content
            return self._parse_llm_response(output, profile)
        except Exception as e:
            return self._create_error_response(profile)

    def _parse_llm_response(self, response: str, profile: list):
        """
        Parses the JSON response from the LLM. Subclasses can override this.
        """
        try:
            # Clean the response first
            cleaned_response = response.strip()
            
            # Try to extract JSON from markdown code blocks
            json_str = self._extract_json(cleaned_response)
            
            # If no JSON found in code blocks, try to find JSON in the response
            if not json_str or json_str == cleaned_response:
                # Look for JSON-like content
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # If no JSON found, try to create a basic response
                    return self._create_basic_response(profile, cleaned_response)
            
            # Parse the JSON
            data = json.loads(json_str)
            actions = data.get("columns", [])
            
            # Fallback for agents that might return a single action dict
            if not actions and isinstance(data, dict):
                return [data]

            for i, col_action in enumerate(actions):
                if i < len(profile):
                    col_action.setdefault("name", profile[i].get("name"))
                    # Pass through missing_count if present in profile
                    if "missing_count" in profile[i]:
                        col_action["missing_count"] = profile[i]["missing_count"]
                col_action.setdefault("reason", "No reason provided by LLM.")
            return actions
            
        except (json.JSONDecodeError, TypeError) as e:
            return self._create_error_response(profile)

    def _create_basic_response(self, profile: list, response: str) -> list:
        """Create a basic response when JSON parsing fails."""
        actions = []
        for col in profile:
            actions.append({
                "name": col.get("name"),
                "suggested_action": "skip",
                "reason": f"LLM response could not be parsed. Raw response: {response[:100]}..."
            })
        return actions

    def _extract_json(self, response: str) -> str:
        """Extracts a JSON string from a markdown code block."""
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def _create_error_response(self, profile: list) -> list:
        """Creates a default 'skip' action when an error occurs."""
        if not profile:
            return [{"name": "unknown", "suggested_action": "skip", "reason": "Agent failed due to an error and no profile was available."}]
            
        return [
            {
                "name": col.get("name"),
                "suggested_action": "skip",
                "reason": "Agent failed to generate a valid suggestion due to an internal error."
            }
            for col in profile
        ]

    def generate_code_from_choice(self, column_name: str, choice: dict) -> str:
        """
        Generates Python code for a chosen cleaning action by invoking the LLM.
        Now logs decisions to memory.
        """
        action = choice.get("suggested_action") or choice.get("action") or choice.get("suggested_strategy")
        if not action or action == "skip":
            return "# No action chosen."
            
        # For feature generation, the formula is the code. No need to call LLM.
        if "formula" in choice:
            code = choice["formula"]
            return code
        
        # For general issues, the code is already generated.
        if "code" in choice:
            code = choice["code"]
            return code

        # --- Special handling for fillna_mean and fillna_median ---
        if action in ["fillna_mean", "fillna_median"]:
            stat_func = "mean" if action == "fillna_mean" else "median"
            
            # Generate intelligent code that preserves data types
            code = f"""
# Handle missing values in {column_name} using {stat_func}
# First convert to numeric to handle any non-numeric values
df[{repr(column_name)}] = pd.to_numeric(df[{repr(column_name)}], errors='coerce')

# Calculate the {stat_func} value
{stat_func}_value = df[{repr(column_name)}].{stat_func}()

# Determine if the column is integer-like (all non-null values are integers)
_non_null = df[{repr(column_name)}].dropna()
_is_integer_like = _non_null.apply(float.is_integer).all() if len(_non_null) > 0 else False

# Fill missing values while preserving the original data type
if _is_integer_like:
    # For integer-like columns, round and cast to Int64
    fill_value = int(round({stat_func}_value))
    df[{repr(column_name)}] = df[{repr(column_name)}].fillna(fill_value).astype('Int64')
else:
    # For float columns, use the {stat_func} value directly
    df[{repr(column_name)}] = df[{repr(column_name)}].fillna({stat_func}_value)
"""
            
            return code

        # --- Special handling for fillna_constant with categoricals ---
        if action == "fillna_constant" and choice.get("constant_value") is not None:
            const_val = choice["constant_value"]
            # The following code will work for both categorical and non-categorical columns
            code = (
                f"if pd.api.types.is_categorical_dtype(df[{repr(column_name)}]):\n"
                f"    if {repr(const_val)} not in df[{repr(column_name)}].cat.categories:\n"
                f"        df[{repr(column_name)}] = df[{repr(column_name)}].cat.add_categories([{repr(const_val)}])\n"
                f"df[{repr(column_name)}] = df[{repr(column_name)}].fillna({repr(const_val)})"
            )
            return code

        # --- Special handling for float64 and int64 conversions ---
        if action in ["float64", "int64", "category", "string"]:
            code = f"df[{repr(column_name)}] = df[{repr(column_name)}].astype('{action}')"
            return code
        if action == "datetime64[ns]":
            code = f"df[{repr(column_name)}] = pd.to_datetime(df[{repr(column_name)}], dayfirst=True, errors='coerce')"
            return code

        # --- Special handling for flag_outliers ---
        if action == "flag_outliers":
            # Always convert to numeric before flagging outliers
            code = (
                f"df[{repr(column_name)}] = pd.to_numeric(df[{repr(column_name)}], errors='coerce')\n"
                f"df[{repr(column_name)}] = df[{repr(column_name)}].where((df[{repr(column_name)}] >= df[{repr(column_name)}].quantile(0.05)) & (df[{repr(column_name)}] <= df[{repr(column_name)}].quantile(0.95)), '')"
            )
            return code

        action_details = f"Action: {action}"
        if choice.get("constant_value") is not None:
            action_details += f", Constant Value: '{choice['constant_value']}'"
        
        reason = choice.get("reason", "No reason provided.")

        code_gen_prompt = self._get_code_generation_prompt_template()
        code_gen_chain = code_gen_prompt | self.llm
        
        response = code_gen_chain.invoke({
            "column_name": column_name,
            "action_details": action_details,
            "reason": reason,
        })
        
        code = response.content.strip()
        cleaned_code = re.sub(r"^```python\n|```$", "", code).strip()
        
        return cleaned_code

    def get_data_dictionary_context(self):
        if self.data_dictionary is None:
            return "No data dictionary provided."
        if isinstance(self.data_dictionary, pd.DataFrame):
            return self.data_dictionary.to_string(index=False)
        return str(self.data_dictionary)