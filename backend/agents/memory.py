import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

class AgentMemory:
    """
    Comprehensive memory system for tracking all aspects of data cleaning sessions.
    Provides context awareness, decision history, and learning capabilities.
    """
    def __init__(self):
        self.conversation_history = []  # LLM interactions and responses
        self.cleaning_decisions = []    # All cleaning actions taken
        self.user_preferences = {}      # User choices and overrides
        self.dataset_context = {}       # Dataset-specific information
        self.agent_performance = {}     # Performance metrics per agent
        self.learning_patterns = {}     # Patterns learned from user behavior
        self.session_metadata = {}      # Session-level information
        
    def log_conversation(self, agent_name: str, prompt: str, response: str, 
                        context: Dict[str, Any] = None):
        """Log LLM conversations for context awareness."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "prompt": prompt,
            "response": response,
            "context": context or {}
        }
        self.conversation_history.append(entry)
        
    def log_cleaning_decision(self, step: int, agent_name: str, column: str, 
                            action: str, reason: str, code: str, 
                            user_override: bool = False, status: str = "success"):
        """Log all cleaning decisions for pattern learning."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "agent": agent_name,
            "column": column,
            "action": action,
            "reason": reason,
            "code": code,
            "user_override": user_override,
            "status": status,
            "dataset_hash": self._get_dataset_hash()
        }
        self.cleaning_decisions.append(entry)
        
    def log_user_preference(self, agent_name: str, column: str, 
                           preference_type: str, value: Any):
        """Track user preferences and overrides."""
        key = f"{agent_name}_{column}_{preference_type}"
        self.user_preferences[key] = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "column": column,
            "preference_type": preference_type,
            "value": value
        }
        
    def set_dataset_context(self, df: pd.DataFrame, data_dictionary: pd.DataFrame = None):
        """Store dataset-specific context information."""
        # Convert data dictionary to serializable format
        data_dict_serializable = None
        if data_dictionary is not None:
            try:
                # Convert to dict and handle non-serializable types
                data_dict_serializable = {}
                for col in data_dictionary.columns:
                    data_dict_serializable[col] = []
                    for val in data_dictionary[col]:
                        if pd.isna(val):
                            data_dict_serializable[col].append(None)
                        elif isinstance(val, (pd.Timestamp, datetime)):
                            data_dict_serializable[col].append(val.isoformat())
                        elif isinstance(val, (list, tuple)):
                            data_dict_serializable[col].append([str(item) for item in val])
                        else:
                            data_dict_serializable[col].append(str(val))
            except Exception as e:
                print(f"Warning: Could not serialize data dictionary: {e}")
                data_dict_serializable = None
        
        self.dataset_context = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "missing_counts": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
            "unique_counts": {str(k): int(v) for k, v in df.nunique().to_dict().items()},
            "data_dictionary": data_dict_serializable,
            "dataset_hash": self._get_dataset_hash()
        }
        
    def get_relevant_history(self, agent_name: str, column: str = None, 
                           limit: int = 10) -> List[Dict]:
        """Get relevant historical decisions for context."""
        relevant = []
        for decision in reversed(self.cleaning_decisions[-limit:]):
            if decision["agent"] == agent_name:
                if column is None or decision["column"] == column:
                    relevant.append(decision)
        return relevant
        
    def get_user_preferences(self, agent_name: str, column: str = None) -> Dict:
        """Get user preferences for specific agent/column combinations."""
        preferences = {}
        for key, pref in self.user_preferences.items():
            if pref["agent"] == agent_name:
                if column is None or pref["column"] == column:
                    preferences[key] = pref
        return preferences
        
    def get_conversation_context(self, agent_name: str, limit: int = 5) -> str:
        """Get recent conversation context for an agent."""
        recent = []
        for conv in reversed(self.conversation_history[-limit:]):
            if conv["agent"] == agent_name:
                recent.append(f"Q: {conv['prompt'][:200]}...\nA: {conv['response'][:200]}...")
        return "\n\n".join(recent)
        
    def learn_patterns(self):
        """Analyze patterns in user behavior and decisions."""
        # Analyze user override patterns
        override_patterns = {}
        for decision in self.cleaning_decisions:
            if decision["user_override"]:
                key = f"{decision['agent']}_{decision['action']}"
                if key not in override_patterns:
                    override_patterns[key] = {"count": 0, "columns": set()}
                override_patterns[key]["count"] += 1
                override_patterns[key]["columns"].add(decision["column"])
                
        # Analyze successful vs failed actions
        success_patterns = {}
        for decision in self.cleaning_decisions:
            key = f"{decision['agent']}_{decision['action']}"
            if key not in success_patterns:
                success_patterns[key] = {"success": 0, "failure": 0}
            if decision["status"] == "success":
                success_patterns[key]["success"] += 1
            else:
                success_patterns[key]["failure"] += 1
                
        self.learning_patterns = {
            "user_overrides": override_patterns,
            "success_rates": success_patterns
        }
        
    def get_enhanced_context(self, agent_name: str, column: str = None) -> Dict[str, Any]:
        """Get comprehensive context for an agent decision."""
        context = {
            "recent_decisions": self.get_relevant_history(agent_name, column, 5),
            "user_preferences": self.get_user_preferences(agent_name, column),
            "conversation_history": self.get_conversation_context(agent_name, 3),
            "dataset_context": self.dataset_context,
            "learning_patterns": self.learning_patterns
        }
        return context
        
    def _get_dataset_hash(self) -> str:
        """Generate a hash for the current dataset for tracking."""
        if not self.dataset_context:
            return "unknown"
        context_str = json.dumps(self.dataset_context, sort_keys=True, default=str)
        return hashlib.md5(context_str.encode()).hexdigest()[:8]
        
    def export_memory(self) -> Dict[str, Any]:
        """Export all memory data for persistence."""
        return {
            "conversation_history": self.conversation_history,
            "cleaning_decisions": self.cleaning_decisions,
            "user_preferences": self.user_preferences,
            "dataset_context": self.dataset_context,
            "agent_performance": self.agent_performance,
            "learning_patterns": self.learning_patterns,
            "session_metadata": self.session_metadata
        }
        
    def import_memory(self, memory_data: Dict[str, Any]):
        """Import memory data from persistence."""
        self.conversation_history = memory_data.get("conversation_history", [])
        self.cleaning_decisions = memory_data.get("cleaning_decisions", [])
        self.user_preferences = memory_data.get("user_preferences", {})
        self.dataset_context = memory_data.get("dataset_context", {})
        self.agent_performance = memory_data.get("agent_performance", {})
        self.learning_patterns = memory_data.get("learning_patterns", {})
        self.session_metadata = memory_data.get("session_metadata", {})
        
    def clear(self):
        """Clear all memory data."""
        self.conversation_history = []
        self.cleaning_decisions = []
        self.user_preferences = {}
        self.dataset_context = {}
        self.agent_performance = {}
        self.learning_patterns = {}
        self.session_metadata = {}
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of memory usage."""
        return {
            "total_conversations": len(self.conversation_history),
            "total_decisions": len(self.cleaning_decisions),
            "total_preferences": len(self.user_preferences),
            "user_overrides": sum(1 for d in self.cleaning_decisions if d["user_override"]),
            "success_rate": sum(1 for d in self.cleaning_decisions if d["status"] == "success") / max(len(self.cleaning_decisions), 1)
        } 