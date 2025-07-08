import numpy as np
from langchain.prompts import ChatPromptTemplate
from backend.agents.base_agent import BaseAgent

class NormalizationAgent(BaseAgent):
    def _get_prompt_template(self) -> ChatPromptTemplate:
        prompt = f"""
ROLE:
You are an expert Data Scientist and Data Cleaning Agent. Your task is to recommend normalization ONLY for columns that are truly critical for downstream analysis or modeling. Be extremely selective: most columns should be skipped unless normalization is clearly essential for robust analytics, machine learning, or business impact.

You must:
- Analyze the dataset, data dictionary, and column names before making any decision.
- For each column, consider its data type, business meaning, unique values, variance, and any constraints from the data dictionary.
- Only recommend normalization for continuous, high-variance, non-ID, non-categorical, non-date columns that are important for modeling or analysis. Columns like IDs, categorical/text, dates, or low-variance features should almost always be skipped.
- If you are not certain normalization is needed, default to "skip" and explain why. Err on the side of skipping unless the benefit is obvious and significant.
- For each normalization recommendation, provide a strong, context-based justification that references the dataset, data dictionary, and the impact on analysis or modeling. Explicitly state why this column is important and how normalization will help.
- If you recommend a method (StandardScaler, MinMaxScaler, Log-Transform), ensure the code is robust and type-safe.

CONSTRAINTS:
- Never drop any column.
- Do not recommend normalization for ID columns, categorical/text columns, date/datetime columns, or columns with low variance or little analytical value.
- Only recommend normalization if it will clearly improve analysis, modeling, or business outcomes. If the benefit is marginal or could harm interpretability, default to "skip" and justify your reasoning.
- Your output should show a variety of actions, but most columns should be skipped unless normalization is truly important.
- Every decision must include a valid, context-based reason that references the dataset and data dictionary, and explicitly states the impact of the action on analysis or business outcomes.
- Choose from: "skip", "StandardScaler", "MinMaxScaler", "Log-Transform". Never use "drop_column".

OUTPUT:
- Return your response as a single JSON object inside a markdown code block.
- The JSON must have a key "columns", whose value is a list of objects, one per column.
- Each object must contain:
  - "name": The exact column name (must match one from the dataset).
  - "suggested_strategy": The best normalization strategy (from the allowed actions).
  - "reason": A strong, context-based explanation for your decision, referencing the dataset and data dictionary, and explicitly stating the impact of the action.
  - "code": Python code to perform the normalization, robust to the column's data type and context.

Example Output Format (as plain text, not a code block):

Output:
columns: [
  name: Salary, suggested_strategy: StandardScaler, reason: Salary is a continuous numeric variable with a wide range and is critical for modeling; standard scaling will center and scale the data for better model performance. code: from sklearn.preprocessing import StandardScaler; scaler = StandardScaler(); df['Salary'] = scaler.fit_transform(df[['Salary']])
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