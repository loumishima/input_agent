from datetime import datetime
from typing import Any, Mapping
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool

from src.prompts.rule_prompt import get_validation_prompt
from src.chains.validator import InputValidator


class ValidatorTools:

    def CHECK_VALID_DATE(self, date: datetime):
        # Verifica se a data é válida.
        return date < datetime.now()

    def CHECK_IF_AVAILABLE(self, date: datetime):
        # TODO: Checar se o dia está disponível
        # HACK: Por enquanto apenas verificar se é um sábado ou domingo
        if date.weekday() == 0 or date.weekday() == 6:
            return False
        return True

    def VALIDATE_TEXTUAL_FIELDS(self, input_dict) -> Mapping[str, Any]:
        validator = InputValidator(get_validation_prompt())
        return validator.validate(input_dict)

    def create_tools(self):

        tool_check_valid_date = Tool.from_function(
            name="check_date",
            description="Verifica se a data é valida e se é uma data posterior a hoje.",
            func=self.CHECK_VALID_DATE,
        )

        tool_availability = Tool.from_function(
            name="check_availability",
            description="Verifica se a data de agendamento está disponível para alocação",
            func=self.CHECK_IF_AVAILABLE,
        )

        tool_validate_text = Tool.from_function(
            name="check_texts",
            description="Verifica se os textos estão ortograficamente e estruturalmente corretos",
            func=self.VALIDATE_TEXTUAL_FIELDS,
        )

        return [tool_check_valid_date, tool_availability, tool_validate_text]


# def criar_tools_da_classe(instancia, metodos):
#     tools = []
#     for metodo_nome in metodos:
#         func = getattr(instancia, metodo_nome)
#         tool = Tool.from_function(
#             name=metodo_nome,
#             description=f"Função {metodo_nome} da classe {instancia.__class__.__name__}",
#             func=func
#         )
#         tools.append(tool)
#     return tools

# # Exemplo
# tools = criar_tools_da_classe(servicos, ["agendar", "cancelar"])
