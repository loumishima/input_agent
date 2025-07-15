from langchain_core.output_parsers import PydanticOutputParser
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

        return (ValidationSchema.isValid, ValidationSchema.whatsWrong)
