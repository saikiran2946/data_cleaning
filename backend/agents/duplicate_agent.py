from backend.agents.base_agent import BaseAgent
import json

class DuplicateAgent(BaseAgent):
    """
    A rule-based agent that specializes in detecting and handling duplicate rows.
    This agent does not use an LLM and instead follows a deterministic approach.
    """
    def _get_prompt_template(self):
        # This agent does not use an LLM, so no prompt is needed.
        return None

    def profile_columns(self):
        """
        Profiles the DataFrame to check for duplicate rows.
        """
        duplicate_rows = int(self.df.duplicated().sum())
        if duplicate_rows == 0:
            return None  # Return None to indicate no duplicates
        
        return {
            "total_rows": len(self.df),
            "total_duplicates": duplicate_rows,
            "columns": ", ".join(list(self.df.columns))
        }

    def generate_actions(self):
        """
        Generates a 'drop_duplicates' action if duplicates are found.
        This method is deterministic and does not call an LLM.
        """
        profile = self.profile_columns()
        
        if profile is None:
            # No duplicates found, return empty list.
            return []
            
        # Duplicates were found, create the action directly.
        num_duplicates = profile["total_duplicates"]
        reason = f"The dataset contains {num_duplicates} duplicate rows. Removing them is essential for accurate analysis."
        
        action = {
            "suggested_action": "drop_duplicates",
            "reason": reason,
            "duplicates_count": num_duplicates
        }
        
        return [action]

    def generate_code_from_choice(self, column_name, choice: dict) -> str:
        action = choice.get("suggested_action")
        if action == "drop_duplicates":
            return "df.drop_duplicates(inplace=True)"
        return "# No action chosen."