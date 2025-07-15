from typing import Any, Mapping
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import Tool

from src.prompts.rule_prompt import get_general_prompt
from src.chains.pipelines import ExtractorPipeline
from src.chains.pipelines import MechanicTagging


class ExtractorTools:

    def extract_data(self, text: str) -> Mapping[str, Any]:
        pipeline = ExtractorPipeline().build_input_pipeline(
            get_general_prompt(),
            parser=PydanticOutputParser(pydantic_object=MechanicTagging),
        )

        return pipeline.invoke({"input": text}).model_dump()

    def create_tools(self):
        tool_extract_text = Tool.from_function(
            name="extract_text",
            description="Extrai informações para criação de um agendamento",
            func=self.extract_data,
        )

        return tool_extract_text
