from typing import List, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command

from src.tools.date import DateValidation
from src.structures.parsers import PessoaInfo
from src.prompts.template import get_prompt_template
from pydantic import BaseModel


from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    dados: PessoaInfo = None
    messages: List[BaseMessage] = None
    llm_output: str = None


llm = ChatOpenAI(model="gpt-4o")
memory = MemorySaver()


def get_static_methods(cls):
    static_methods = []
    for _, attr in cls.__dict__.items():
        if isinstance(attr, staticmethod):
            static_methods.append(attr.__func__)  # pega a função "real"
    return static_methods


prompt = SystemMessage(
    content="""
        Você é um extrator de informações de serviços automotivos
        Extraia as seguintes informações do texto abaixo.
        Caso algo falte, por favor adicione essa info no campo 'erro', note
        que alguns valores são opcionais como ano do carro:

        Também valide se a data do agendamento é válida e se os serviços estão disponíveis
        ao cliente.

        Salve os dados estruturados para que o próximo agente consiga avançar no agendamento
                       """
)

agente_extrator = create_react_agent(
    llm,
    tools=get_static_methods(DateValidation),
    prompt=prompt,
    response_format=PessoaInfo,
)


def extract(state: State):
    result = agente_extrator.invoke({"messages": state["messages"]})

    state["messages"] = result["messages"]
    state["dados"] = result["structured_response"]
    state["llm_output"] = result["messages"][-1].content

    return state


def build_workflow():
    builder = StateGraph(State)
    builder.add_node("extrator", extract)

    builder.set_entry_point("extrator")
    builder.add_edge("extrator", END)

    return builder.compile(checkpointer=memory)


# agente_cotador = create_react_agent(llm, tools=None, prompt=None)


if __name__ == "__main__":

    config = {"configurable": {"thread_id": "test-thread"}}
    workflow = build_workflow()
    # Teste

    messages = [
        HumanMessage(
            content="Tenho um ford Ka 2004 para trocar o óleo e quero que seja no dia 21/07/2025"
        )
    ]

    result = workflow.invoke({"messages": messages}, config)
    for m in result["messages"]:
        m.pretty_print()

    # print(messages["__interrupt__"])
    # Output:
    # Interrupt(value={'question': 'Do you approve the following output?', 'llm_output': 'This is the generated output.'}, ...)

    # Simulate resuming with human input
    # To test rejection, replace resume="approve" with resume="reject"
    # final_result = workflow.invoke(Command(resume="approve"), config=config)
    from pprint import pprint

    pprint(result)
    print(result.keys())
    # print(result["llm_output"])
    # print(messages.structured_response)
