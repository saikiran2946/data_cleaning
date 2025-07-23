from backend.agents.missing_value_agent import MissingValueAgent
from backend.agents.outlier_agent import OutlierAgent
from backend.agents.duplicate_agent import DuplicateAgent
from backend.agents.data_type_agent import DataTypeAgent
from backend.agents.normalization_agent import NormalizationAgent
from backend.agents.value_standardization_agent import ValueStandardizationAgent
from backend.agents.feature_generation_agent import FeatureGenerationAgent
from backend.agents.validating_agent import ValidatingAgent
from backend.agents.general_issue_agent import GeneralIssueAgent
from utils.openai_client import llm, invoke_with_tracking
import json
import numpy as np

class RootAgent:
    """
    The main orchestrator that holds instances of all specialized cleaning agents.
    """
    def __init__(self, df, data_dictionary=None):
        self.df = df
        self.data_dictionary = data_dictionary
        
        agent_classes = {
            "Data Types": DataTypeAgent,
            "Missing Values": MissingValueAgent,
            "Duplicates": DuplicateAgent,
            "Outliers": OutlierAgent,
            "Normalization": NormalizationAgent,
            "Value Standardization": ValueStandardizationAgent,
            #"Feature Generation": FeatureGenerationAgent,
            "Validation": ValidatingAgent,
            #"General Issue": GeneralIssueAgent,
        }
        
        # Initialize all agents
        self.agents = {}
        for name, cls in agent_classes.items():
            agent = cls(self.df, self.data_dictionary)
            self.agents[name] = agent
        
        self.available_agent_names = list(agent_classes.keys())

    def get_agent(self, agent_name: str):
        """Returns an instance of the requested agent."""
        return self.agents.get(agent_name)

    def get_llm_cleaning_plan(self):
        """
        Uses the LLM to suggest the optimal sequence of agent names (and reasons) to invoke, based on the dataset and data dictionary.
        """
        # Create a detailed dataset profile for the prompt
        profile_parts = []
        profile_parts.append("DATASET PROFILE:")
        profile_parts.append(f"Shape: {self.df.shape}")
        # Always include full dtypes listing
        profile_parts.append("\nColumns and Data Types:")
        profile_parts.append(self.df.dtypes.to_string())
        # List numeric columns
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        profile_parts.append(f"\nNumeric columns: {numeric_cols}")
        # Numeric columns summary
        if numeric_cols:
            profile_parts.append("\nNumeric Columns Summary (describe):")
            profile_parts.append(self.df[numeric_cols].describe().T.to_string())
            # For each numeric column, show a few non-null sample values
            for col in numeric_cols:
                non_null_vals = self.df[col].dropna().head(5).tolist()
                profile_parts.append(f"Sample values for {col}: {non_null_vals}")
        else:
            profile_parts.append("\nNo numeric columns found.")
        # Missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            profile_parts.append("\nMissing Values per Column:")
            profile_parts.append(missing_values[missing_values > 0].to_string())
        else:
            profile_parts.append("\nMissing Values: None found.")
        # Duplicates
        if self.df.duplicated().sum() > 0:
            profile_parts.append(f"\nDuplicate Rows: Found {self.df.duplicated().sum()} duplicate rows.")
        else:
            profile_parts.append("\nDuplicate Rows: None found.")
        # Sample rows
        profile_parts.append("\nSample Rows:")
        profile_parts.append(self.df.head(5).to_string(index=False))
        # Unique value samples for categorical/text columns
        for col in self.df.select_dtypes(include=['object', 'category']):
            unique_vals = self.df[col].dropna().unique()
            profile_parts.append(f"Column '{col}' unique values (sample): {list(unique_vals[:10])}")
        dataset_profile = "\n".join(profile_parts)

        # Describe agent capabilities for the prompt
        agent_descriptions = {
            "Data Types": "Fixes incorrect column data types (e.g., text instead of numbers, numbers instead of dates).",
            "Missing Values": "Handles missing data through various imputation strategies (mean, median, mode, constant).",
            "Duplicates": "Removes duplicate rows from the dataset.",
            "Outliers": "Manages extreme or anomalous values in numeric columns.",
            "Normalization": "Scales numeric data for modeling (e.g., StandardScaler, MinMaxScaler).",
            "Value Standardization": "Corrects inconsistent categorical values (e.g., 'USA', 'U.S.A.', 'United States').",
            #"Feature Generation": "Creates new features from existing ones to improve analytical power.",
            "Validation": "Performs a final, comprehensive check to ensure all data quality issues are resolved.",
            #"General Issue": "Handles more complex, user-described issues that don't fit other categories."
        }
        available_agents_text = "\n".join([f"- {name}: {desc}" for name, desc in agent_descriptions.items() if name in self.available_agent_names])

        prompt = f'''
ROLE:
You are a master AI Data Cleaning Strategist. Your responsibility is to conduct an exhaustive analysis of a dataset and formulate a precise, efficient, and logically sound cleaning plan.

CRITICAL THINKING:
1.  **Examine the Full Picture:** Scrutinize the entire DATASET PROFILE and DATA DICTIONARY. Cross-reference them to find not just obvious issues (like missing values) but also subtle ones (e.g., a column is `int64` but the data dictionary says it represents categories and should be `object`).
2.  **MANDATORY AGENTS:** You MUST always include all of the following agents in the cleaning plan, in this order: Data Types, Missing Values, Duplicates, Outliers, Normalization, Value Standardization, Validation.
3.  **Context-Aware Decisions:** For each agent, analyze the dataset and data dictionary to determine if any action is needed. If action is needed, provide a clear, data-driven reason. If no action is needed, state 'No action needed for this dataset.'
4.  **Justify Every Step:** Your reason for each agent must be grounded in specific observations from the profile (e.g., 'The Salary column has a max value of 500000, which is a potential outlier, so the Outlier agent is needed.' or 'No missing values found, so no action needed for Missing Values agent.').
5.  **Optimize the Order:** The sequence of operations is critical. The required order is: Data Types, Missing Values, Duplicates, Outliers, Normalization, Value Standardization, Validation. Do not change this order.

ACTION:
- For each agent, always include it in the plan, in the required order.
- For each agent, provide a key 'agent_name' and a key 'reason'.
- If no action is needed for an agent, the reason should be 'No action needed for this dataset.'
- The plan must be an ordered sequence of steps, one for each agent.
- The 'Validation' agent must ALWAYS be the final step.

AVAILABLE AGENTS AND THEIR CAPABILITIES:
{available_agents_text}

DATA DICTIONARY:
{self.data_dictionary.to_string(index=False) if self.data_dictionary is not None else 'No data dictionary provided.'}

{dataset_profile}

OUTPUT:
- Return your response as a single JSON object inside a markdown code block.
- The JSON must have a key "plan", which is a list of objects, one per step.
- Each object must contain:
  - "agent_name": The exact agent name from the list.
  - "reason": A clear, evidence-based explanation for why this agent is needed for this dataset and at this specific step.

Example Output Format (as plain text, not a code block):
{{
  "plan": [
    {{
      "agent_name": "Data Types",
      "reason": "The 'Age' column is an object but should be numeric, and 'JoinDate' is an object but should be a datetime. This must be fixed first as subsequent numeric and date-based agents depend on it."
    }},
    {{
      "agent_name": "Missing Values",
      "reason": "The 'Age' and 'Salary' columns have missing values that need to be imputed before outlier detection or normalization can be performed accurately."
    }},
    {{
      "agent_name": "Validation",
      "reason": "Perform a final validation to ensure all identified data quality issues have been resolved and the dataset is clean."
    }}
  ]
}}
'''
        response = invoke_with_tracking(llm.invoke, prompt, agent_name="CleaningPlan")
        
        try:
            content = response.content.strip()
            # Try to extract JSON if wrapped in markdown
            if content.startswith("```json"):
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            # The prompt now asks for a "plan" key
            parsed_json = json.loads(content)
            plan = parsed_json.get("plan", [])

            # --- Ensure all agents are present in the required order ---
            required_agents = [
                ("Data Types", "Fixes incorrect column data types (e.g., text instead of numbers, numbers instead of dates)."),
                ("Missing Values", "Handles missing data through various imputation strategies (mean, median, mode, constant)."),
                ("Duplicates", "Removes duplicate rows from the dataset."),
                ("Outliers", "Manages extreme or anomalous values in numeric columns."),
                ("Normalization", "Scales numeric data for modeling (e.g., StandardScaler, MinMaxScaler)."),
                ("Value Standardization", "Corrects inconsistent categorical values (e.g., 'USA', 'U.S.A.', 'United States')."),
                ("Validation", "Performs a final, comprehensive check to ensure all data quality issues are resolved.")
            ]
            # Build a dict for quick lookup
            plan_dict = {step.get("agent_name"): step for step in plan}
            final_plan = []
            for agent_name, default_reason in required_agents:
                if agent_name in plan_dict:
                    step = plan_dict[agent_name]
                    # Ensure reason is present
                    if not step.get("reason"):
                        step["reason"] = default_reason
                    final_plan.append(step)
                else:
                    final_plan.append({"agent_name": agent_name, "reason": "No action needed for this dataset."})
            return final_plan
        except Exception as e:
            # Fallback: just run all agents in default order
            fallback_plan = [
                {"agent_name": name}
                for name in self.available_agent_names
            ]
            # --- Ensure Validation is always last in fallback ---
            validation_steps = [step for step in fallback_plan if step.get("agent_name") == "Validation"]
            other_steps = [step for step in fallback_plan if step.get("agent_name") != "Validation"]
            fallback_plan = other_steps + validation_steps
            return fallback_plan

    def get_cleaning_plan(self):
        return self.get_llm_cleaning_plan()

    def get_agent_recommendations(self, column_name: str = None) -> dict:
        """Get agent-specific recommendations based on memory."""
        recommendations = {}
        
        for agent_name, agent in self.agents.items():
            context = {}
            
            # Analyze recent decisions for this agent
            recent_decisions = context.get("recent_decisions", [])
            if recent_decisions:
                success_rate = sum(1 for d in recent_decisions if d["status"] == "success") / len(recent_decisions)
                recommendations[agent_name] = {
                    "success_rate": success_rate,
                    "recent_actions": [d["action"] for d in recent_decisions[:3]],
                    "user_preferences": len(context.get("user_preferences", []))
                }
            else:
                recommendations[agent_name] = {
                    "success_rate": 0.0,
                    "recent_actions": [],
                    "user_preferences": 0
                }
        
        return recommendations