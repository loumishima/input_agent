from langchain_core.messages import SystemMessage
from src.structures.states import State
from src.agents.extractor_agent import ExtractorAgent
from langgraph.graph import StateGraph, END

from src.prompts.template import get_prompt_template

from dotenv import load_dotenv

load_dotenv()


def run_extractor(state):
    agent = ExtractorAgent(
        model_name="gpt-4o", state=state, prompt=get_prompt_template()
    )
    return agent.execute()


def workflow():

    workflow_builder = StateGraph(State)  # seu TypedDict State

    workflow_builder.add_node("extract", run_extractor)
    workflow_builder.set_entry_point("extract")
    workflow_builder.add_edge("extract", END)

    return workflow_builder.compile()


if __name__ == "__main__":
    initial_state = State(
        query="Oi, nasci em 20 de março de 1990.",
        messages=[],
        category=None,
        answer=None,
        next_tool=None,
    )
    workflow_compiled = workflow()
    final_state = workflow_compiled.invoke(initial_state)
    print("Resposta da LLM:", final_state["answer"])
    print("Histórico de mensagens:", [m.content for m in final_state["messages"]])

    print(type(final_state))
    print(type(final_state["answer"]))
    while final_state["answer"].erro:
        print(final_state["answer"].erro)

        final_state["query"] = input("Complete as infos restantes")
        final_state = workflow_compiled.invoke(final_state)

        print(final_state["answer"])
