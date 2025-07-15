from typing import Optional, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.memory import BaseMemory
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel

from src.agents.agent import start_agent
from src.chains.pipelines import MechanicTagging, ExtractorPipeline
from src.chains.validator import InputValidator
from src.models.llm import get_llm_model
from src.prompts.rule_prompt import (
    get_general_prompt,
    get_validation_prompt,
    get_confirmation_message_prompt,
)


def build_message(llm, prompt, parser):
    pipeline = prompt | llm | parser

    return pipeline


if __name__ == "__main__":

    agent = start_agent()

    resposta = agent.run("Por favor, agende troca de óleo para o João no dia 22/07.")
    print(resposta)
