from datetime import datetime
from langchain_core.tools import tool

from langgraph.types import interrupt


class DateValidation:

    @staticmethod
    @tool("is_weekend")
    def isWeekendTool(date: datetime):
        """
        Verifica se o dia do agendamento é um sábado ou domingo, se sim
        avisa ao cliente que a oficina não funciona.

        """
        if date.weekday() >= 5:
            return "A oficina não funciona nos fins de semana"
        else:
            return "A oficina está aberta nesse dia"

    @staticmethod
    @tool("is_schedule_valid")
    def validateScheduleDateTool(date: datetime):
        """
        Verifica se a data é anterior ao dia de hoje

        """
        if date.replace(tzinfo=None) < datetime.now():
            return "A data para o servico precisa ser depois de hoje"

    @staticmethod
    @tool("schedule_on_calendar")
    def scheduleOnCalendar(date: datetime, state):
        """
        Agenda no calendario para a data solicitada se os dados forem válidos.
        Solicita a confirmação pelo usuário
        """

        response = interrupt(f"A data {date} esta disponível, deseja agendar? (S/n)")
        if response["type"] == "sim":
            pass
        elif response["type"] == "não":
            hotel_name = response["args"]["hotel_name"]
        else:
            raise ValueError(f"Valor inválido: {response['type']}")

        return f"{state} agendado na {date} no calendário"
