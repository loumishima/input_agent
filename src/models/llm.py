from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_llm_model(temperature=0.2):

    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
