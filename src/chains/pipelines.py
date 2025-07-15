from pydantic import BaseModel
from typing import Optional, Sequence
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from src.models.llm import get_llm_model


class MechanicTagging(BaseModel):
    client: Optional[str]
    car_model: Optional[str]
    car_year: Optional[int]
    visit_date: Optional[str]
    services: Sequence[str]


def build_input_pipeline(llm, prompt, parser):
    pipeline = (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )
    return pipeline


def build_message(llm, prompt, parser):
    pipeline = prompt | llm | parser

    return pipeline


def build_parallel_pipeline(llm, prompts, parsers, validation_prompt=None):
    return RunnableParallel(
        {
            "dados": build_input_pipeline(
                llm, prompts[0], parsers[0], validation_prompt
            ),
            "confirmacao": build_message(llm, prompts[1], parsers[1]),
        }
    )


class ExtractorPipeline:
    def __init__(self):
        self.llm = get_llm_model()

    def build_confirmation_message(self, prompt, parser):
        return prompt | self.llm | parser

    def build_input_pipeline(self, prompt, parser):
        return (
            prompt.partial(format_instructions=parser.get_format_instructions())
            | self.llm
            | parser
        )

    def build_parallel_pipeline(self, prompts, parsers):
        return RunnableParallel(
            {
                "dados": self.build_input_pipeline(prompts[0], parsers[0]),
                "confirmacao": self.build_confirmation_message(prompts[1], parsers[1]),
            }
        )
