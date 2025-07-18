from typing import Optional
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ExtractedData(BaseModel):
    nome: str
    data_nascimento: str
    memoria: str
    erros: Optional[str]


class PessoaInfo(BaseModel):
    nome: str = Field(..., description="Nome completo da pessoa")
    data_nascimento: str = Field(..., description="Data de nascimento da pessoa")
    erro: str = Field(
        ...,
        description="Explique se alguma informação está faltando ou incorreta. Caso tudo esteja correto, deixe esse campo vazio.",
    )
