from langchain_core.messages import SystemMessage, HumanMessage
from src.structures.states import State
from src.agents.extractor_agent import ExtractorAgent
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from src.structures.parsers import PessoaInfo

from src.tools.date import DateValidation
from src.prompts.template import get_prompt_template

from dotenv import load_dotenv

load_dotenv()

tools = [
    DateValidation.isWeekendTool,
    DateValidation.validateScheduleDateTool,
    DateValidation.scheduleOnCalendar,
]
llm = ChatOpenAI(model="gpt-4o")

prompt = SystemMessage(
    content="""
        Você é um agendador de serviços automotivos
        Extraia as seguintes informações do texto abaixo.
        Caso algo falte, por favor adicione essa info no campo 'erro', note
        que alguns valores são opcionais como ano do carro:

        Também valide se a data do agendamento é válida e se os serviços estão disponíveis
        ao cliente.
                       """
)

memory = MemorySaver()
agent = create_react_agent(
    llm, tools=tools, prompt=prompt, checkpointer=memory, response_format=PessoaInfo
)
config = {"configurable": {"thread_id": "test-thread"}}


if __name__ == "__main__":
    messages = [
        HumanMessage(
            content="Tenho um ford Ka 2004 para trocar o óleo e quero que seja no dia 21/07/2025"
        )
    ]
    messages = agent.invoke({"messages": messages}, config)
    for m in messages["messages"]:
        m.pretty_print()

    print(messages["structured_response"])


def workflow():

    workflow_builder = StateGraph(State)  # seu TypedDict State

    workflow_builder.add_node("extract", execute_agent)
    workflow_builder.set_entry_point("extract")
    workflow_builder.add_edge("extract", END)

    return workflow_builder.compile()


# if __name__ == "__main__":
#     initial_state = State(
#         query="Oi, nasci em 20 de março de 1990.",
#         messages=[],
#         category=None,
#         answer=None,
#         next_tool=None,
#     )
#     workflow_compiled = workflow()
#     final_state = workflow_compiled.invoke(initial_state)
#     print("Resposta da LLM:", final_state["answer"])
#     print("Histórico de mensagens:", [m.content for m in final_state["messages"]])

#     print(type(final_state))
#     print(type(final_state["answer"]))
#     while final_state["answer"].erro:
#         print(final_state["answer"].erro)

#         final_state["query"] = input("Complete as infos restantes")
#         final_state = workflow_compiled.invoke(final_state)

#         print(final_state["answer"])
