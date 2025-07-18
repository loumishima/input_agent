from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.structures.parsers import PessoaInfo


def get_prompt_template():

    parser = PydanticOutputParser(pydantic_object=PessoaInfo)

    prompt = PromptTemplate(
        template="""
        Extraia as seguintes informações do texto abaixo.
        Caso algo falte, por favor adicione essa info no campo 'erro':

        Histórico de mensagens: {history}

        Texto: {input}

        {format_instructions}

        """,
        input_variables=["input", "history"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    return prompt
