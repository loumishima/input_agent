from typing import Optional, Sequence
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ExtractedData(BaseModel):
    nome: str
    data_nascimento: str
    memoria: str
    erros: Optional[str]


class PessoaInfo(BaseModel):
    # nome: str = Field(..., description="Nome completo da pessoa")
    # data_nascimento: str = Field(..., description="Data de nascimento da pessoa")
    carro: Optional[str] = Field(..., description="Modelo do carro")
    ano_carro: Optional[str] = Field(..., description="Ano do modelo do carro")
    data_agendamento: Optional[str] = Field(
        ..., description="Data para o agendamento do serviço no carro"
    )
    servicos: Optional[Sequence[str]] = Field(
        ..., description="Servicos a serem realizados no carro"
    )
    erro: Optional[str] = Field(
        ...,
        description="Explique se alguma informação está faltando ou incorreta. Caso tudo esteja correto, deixe esse campo vazio.",
    )
