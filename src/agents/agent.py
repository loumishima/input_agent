from src.models.llm import get_llm_model
from langchain.agents import initialize_agent, AgentType

from src.tools.database import DatabaseManagerTools
from src.tools.extractor import ExtractorTools
from src.tools.validator import ValidatorTools


def start_agent():
    all_tools = []

    all_tools.extend(DatabaseManagerTools().create_tools())
    all_tools.extend(ExtractorTools().create_tools())
    all_tools.extend(ValidatorTools().create_tools())

    agent = initialize_agent(
        tools=all_tools,
        llm=get_llm_model(),
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    return agent
