from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from src.structures.states import State

from langchain_core.output_parsers import PydanticOutputParser
from src.structures.parsers import PessoaInfo


class ExtractorAgent:

    def __init__(self, model_name, state: State, prompt: SystemMessage = None):
        self.llm = self.create_model(model_name)
        self.state = state
        self.parser = PydanticOutputParser(pydantic_object=PessoaInfo)

        self.prompt = prompt
        if not self.prompt:
            self.prompt = SystemMessage(
                content="Você é um assistente inteligente, responda de forma clara e precisa."
            )

    def save_message(self, message: BaseMessage):
        self.state.messages.append(message)

    def format_prompt(self):

        history_text = "\n".join([msg.content for msg in self.state.messages])
        formatted_prompt = self.prompt.format(
            input=self.state.query, history=history_text
        )

        return formatted_prompt

    def execute(self) -> State:

        formatted_prompt = self.format_prompt()

        self.save_message(HumanMessage(content=self.state.query))

        response = self.llm.invoke(formatted_prompt)

        self.state.answer = self.parser.parse(response.content)

        self.save_message(AIMessage(content=response.content))

        return self.state

    def create_model(self, model_name):
        return ChatOpenAI(model=model_name)

    def use_tools(self, model_name):
        pass
