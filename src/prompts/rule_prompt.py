from langchain_core.prompts import PromptTemplate


def get_general_prompt():

    return PromptTemplate.from_template(
        """
        Você é um assistente de oficina mecânica.
        Extraia as seguintes informações do texto:
        - Nome do cliente (se houver)
        - Modelo do carro (se houver)
        - Data (se houver)
          - Formate a data com hífen
          - Adicione o ano atual ao fim da data se necessário
        - Lista de serviços mencionados (se houver)

        Responda no seguinte formato JSON:
        {format_instructions}

        Texto: {input}
        """
    )


def get_confirmation_message_prompt():
    return PromptTemplate.from_template(
        "Crie uma mensagem simpática de confirmação de serviço para o seguinte cliente: {input}"
    )


def get_validation_prompt():
    return PromptTemplate.from_template(
        """
        Você receberá um conjunto de dados e validará se esses dados são validos,
        retornando True se estiver tudo ok e False se estiver algo errado, além de uma
        explicação do que estiver errado para o usuário corrigir
        seguindo essas regras:

        - Verifique a data, seguindo o padrão "dd-mm-yyyy"
        - Verifique se o modelo do carro existe.
        - Verifique se os serviços estão disponíveis para carros
         - Serviços disponíveis: Mecânica e Hidráulica
         - Bloqueie qualquer tópico que não seja dessas duas áreas

        Retorne nesse formato JSON:
        {format_instructions}

        Dados em JSON: {input}

        """
    )
