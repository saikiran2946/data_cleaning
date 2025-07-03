import numpy as np
from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent

class NormalizationAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = f"""
ROLE:
You are an advanced, context-aware Data Cleaning Agent specializing in normalization for tabular datasets.

You must always:
- Analyze the dataset, data dictionary, and column names before making any decision.
- Infer the intended data type and business meaning for each column from the data, business context, and data dictionary, not just the pandas dtype.
- Justify every normalization action and code with reference to the dataset's context, business meaning, and data dictionary.

ACTION:
For every column, you must:
- Analyze the column's data type, business meaning, unique values, and any constraints from the data dictionary.
- Critically evaluate the necessity and impact of normalization for each column. Only recommend normalization if it is clearly beneficial for analysis, modeling, or business outcomes. If the benefit is marginal or uncertain, default to "skip" and explain why.
- Provide a context-based reason for your choice, referencing the dataset, data dictionary, and column meaning.
- If you recommend a specific method (e.g., StandardScaler, MinMaxScaler, Log-Transform), ensure the code is type-safe and will not cause errors for non-numeric or mixed-type columns.

CONSTRAINTS:
- Never drop any column, regardless of distribution or context.
- Use the data dictionary and dataset profile for all decisions; do not rely on hardcoded rules.
- Make decisions dynamically and contextually for each column, always considering the intended data type and business meaning.
- The "suggested_strategy" must be the best fit for each column; do not use the same action for all columns.
- For categorical or text columns, do not apply numeric normalization methods; explain why normalization is not applicable.
- For date/datetime columns, only apply normalization logic if it makes sense in the business context, and ensure the code is robust to string or datetime types.
- For numeric columns, ensure the code is robust to mixed types (e.g., strings, NaNs) and does not cause type errors.
- Only recommend normalization if it will clearly improve analysis, modeling, or business outcomes. If the benefit is marginal or could harm interpretability or downstream use, default to "skip" and justify your reasoning.
- Your output should show a variety of actions, reflecting the true context and needs of each column.
- Every decision must include a valid, context-based reason that references the dataset and data dictionary, and explicitly states the impact of the action on analysis or business outcomes.
- Choose from: "skip", "StandardScaler", "MinMaxScaler", "Log-Transform". Never use "drop_column".

OUTPUT:
- Return your response as a single JSON object inside a markdown code block.
- The JSON must have a key "columns", whose value is a list of objects, one per column.
- Each object must contain:
  - "name": The exact column name (must match one from the dataset).
  - "suggested_strategy": The best normalization strategy (from the allowed actions).
  - "reason": A clear, context-based explanation for your decision, referencing the dataset and data dictionary, and explicitly stating the impact of the action.
  - "code": Python code to perform the normalization, robust to the column's data type and context.

Example Output Format (as plain text, not a code block):

Output:
columns: [
  name: Salary, suggested_strategy: StandardScaler, reason: Salary is a continuous numeric variable with a wide range; standard scaling will center and scale the data for better model performance. code: from sklearn.preprocessing import StandardScaler; scaler = StandardScaler(); df['Salary'] = scaler.fit_transform(df[['Salary']])
  name: Department, suggested_strategy: skip, reason: Department is categorical; normalization is not applicable and would harm interpretability. code: # No action chosen.
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
            if self.df[col].nunique() <= 10 or 'id' in col.lower():
                continue
            profile.append({
                "name": col,
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "skew": float(self.df[col].skew()),
            })
        return profile

    def generate_code_from_choice(self, column_name, choice: dict) -> str:
        action = choice.get("suggested_strategy") or choice.get("suggested_action")
        if not action or action == "skip":
            return "# No action chosen."
        if action == "StandardScaler":
            return f"df['{column_name}'] = (df['{column_name}'] - df['{column_name}'].mean()) / df['{column_name}'].std()"
        if action == "MinMaxScaler":
            return f"df['{column_name}'] = (df['{column_name}'] - df['{column_name}'].min()) / (df['{column_name}'].max() - df['{column_name}'].min())"
        if action == "Log-Transform":
            return f"df['{column_name}'] = np.log1p(df['{column_name}'])"
        return "# No valid normalization action chosen." 