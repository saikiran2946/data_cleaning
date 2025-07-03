from langchain.tools import BaseTool

class AgentTool(BaseTool):
    """A tool that wraps another agent to be used by a parent agent."""
    class Config:
        arbitrary_types_allowed = True

    agent: object  # The agent to wrap

    def __init__(self, agent_name: str, agent_instance: object, description: str):
        super().__init__(name=agent_name, description=description, agent=agent_instance)

    def _run(self, query: str) -> str:
        """
        Invokes the underlying agent with a query and returns the result.
        The input `query` should be the profile of the data for the agent to process.
        """
        # The wrapped agent's main entry point is now 'generate_actions'
        # which internally calls the LangChain executor.
        return self.agent.generate_actions() 