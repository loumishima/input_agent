from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
from langchain_core.messages import BaseMessage
from langchain.tools import Tool
from src.structures.parsers import PessoaInfo


class State(BaseModel):
    query: str
    answer: Optional[PessoaInfo]
    messages: List[BaseMessage]
    tool_callers: List[Tool]
