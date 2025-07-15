from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional

from src.models.llm import get_llm_model


class validationMessage(BaseModel):
    isValid: bool
    whatsWrong: Optional[str]


class InputValidator:
    def __init__(self, prompt):
        self.llm = get_llm_model()
        self.prompt = prompt
        self.parser = PydanticOutputParser(pydantic_object=validationMessage)

    def validate_pipeline(self):

        return (
            self.prompt.partial(
                format_instructions=self.parser.get_format_instructions()
            )
            | self.llm
            | self.parser
        )

    def validate(self, input):

        ValidationSchema = self.validate_pipeline().invoke({"input": input})

        return ValidationSchema.model_dump()

        # if not ValidationSchema.isValid:
        #     return ValidationSchema.whatsWrong

        # # mande uma mensagem de sucesso na escrita
        # prompt = PromptTemplate.from_template(
        #     """
        #     Gere uma mensagem de resposta com o nome do cliente,
        #     agradecendo pela escolha e mostre uma
        #     confirmação da reserva:

        #     {input}
        #     """
        # )
        # chain = prompt | self.llm | StrOutputParser()

        # return chain.invoke({"input": input})
