from typing import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

from src.structures.parsers import PessoaInfo


from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    messages: List[BaseMessage]
    user_input: str = None
    llm_response: str = None
    dados: PessoaInfo = None
    error: str = None
    approved: bool = False


llm = ChatOpenAI(model="o4-mini")
memory = MemorySaver()


def get_static_methods(cls):
    static_methods = []
    for _, attr in cls.__dict__.items():
        if isinstance(attr, staticmethod):
            static_methods.append(attr.__func__)  # pega a função "real"
    return static_methods


def intro(state: State):
    result = llm.invoke(
        [
            SystemMessage(
                content="""
                Gere uma mensagem de boas vindas a uma oficina, indicando que ele tem que adicionar:
                - modelo do veiculo
                - Ano do veiculo (Opcional)
                - Servicos a serem feitos
                - Data do agendamento, indicando funcionamento de segunda a sexta
                """
            )
        ]
    )

    print(result.content)


llm_extractor = llm.with_structured_output(PessoaInfo)


def extract(state: State):

    if not state.get("user_input"):
        msg = interrupt({})
    else:
        msg = state["user_input"]
    print(msg)
    result = llm_extractor.invoke(msg)
    return {
        "user_input": msg,
        "dados": result,
        "error": result.error,
    }


def validador(state: State):
    print("================= Validacao ===================")
    from pprint import pprint

    pprint(state)
    if state["error"]:
        extra_info = interrupt({"question": state["error"]})
        return {"user_input": state["user_input"] + " " + extra_info}


def confirm(state: State):
    text = "Por favor confirme os dados abaixo, está tudo correto? (s/n)"
    print(text)
    confirmation = interrupt(
        {
            "question": text,
            "dados": state["dados"],
        }
    )

    if confirmation == "s":
        approved = True
        dados = state["user_input"]
    else:
        approved = False
        dados = None

    return {"approved": approved, "user_input": dados}


def route_validacao(state: State):
    if state["error"]:
        return "Fix"
    else:
        return "Confirm"


def route_dados(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["approved"]:
        return "Accepted"
    return "Rejected"


def build_workflow():
    builder = StateGraph(State)
    builder.add_node("intro", intro)
    builder.add_node("extrair", extract)
    builder.add_node("validar", validador)
    builder.add_node("confirmar", confirm)

    builder.set_entry_point("intro")
    builder.add_edge("intro", "extrair")
    builder.add_edge("extrair", "validar")
    builder.add_conditional_edges(
        "validar",
        route_validacao,
        {  # Name returned by route_joke : Name of next node to visit
            "Fix": "extrair",
            "Confirm": "confirmar",
        },
    )
    builder.add_conditional_edges(
        "confirmar", route_dados, {"Accepted": END, "Rejected": "extrair"}
    )

    return builder.compile(checkpointer=memory)


# agente_cotador = create_react_agent(llm, tools=None, prompt=None)


if __name__ == "__main__":

    values_to_input = ["gostaria de trocar o óleo para o dia 22/07/2025"]

    config = {"configurable": {"thread_id": "test-thread"}}
    workflow = build_workflow()

    result = workflow.invoke({}, config)

    while result.get("__interrupt__"):
        try:
            message = values_to_input.pop(0)
        except IndexError:
            message = input()
        result = workflow.invoke(Command(resume=message), config=config)

    print(result)
