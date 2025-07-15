from typing import Optional, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.memory import BaseMemory
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel

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

    prompt = get_general_prompt()

    prompt_confirm = get_confirmation_message_prompt()

    parser = PydanticOutputParser(pydantic_object=MechanicTagging)
    parser_confirm = StrOutputParser()

    prompt_list = [prompt, prompt_confirm]
    parser_list = [parser, parser_confirm]

    extractor = ExtractorPipeline()
    # pipeline = extractor.build_parallel_pipeline(prompt_list, parser_list)

    pipeline = extractor.build_input_pipeline(prompt, parser)

    texto_oficina = "Ontem, o cliente Marcos trouxe o Honda Civic 2009 para alinhamento e troca de óleo. Quero agendar o serviço para o dia 12/07."

    resultado = pipeline.invoke({"input": texto_oficina})

    print(resultado)
    print(type(resultado))

    validator = InputValidator(get_validation_prompt())

    print(validator.validate(resultado.model_dump()))
