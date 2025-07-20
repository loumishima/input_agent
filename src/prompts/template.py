from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.structures.parsers import PessoaInfo


def get_prompt_template():

    parser = PydanticOutputParser(pydantic_object=PessoaInfo)

    prompt = PromptTemplate(
        template="""
        Você é um agendador de serviços automotivos
        Extraia as seguintes informações do texto abaixo.
        Caso algo falte, por favor adicione essa info no campo 'erro', note
        que alguns valores são opcionais como ano do carro:

        Também valide se a data do agendamento é válida e se os serviços estão disponíveis
        ao cliente.

        Texto: {messages}

        {format_instructions}

        """,
        input_variables=["messages"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    return prompt
