from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from langchain_core.messages import BaseMessage
from src.structures.parsers import PessoaInfo


class State(BaseModel):
    query: str
    category: Optional[str]
    answer: Optional[PessoaInfo]
    messages: List[BaseMessage]
