from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()


def busca_dados_estudante(estudante):
    dados = pd.read_csv("./documentos/estudantes.csv")
    dados_estudante = dados[dados["USUARIO"] == estudante]
    if dados_estudante.empty:
        return {}
    return dados_estudante.iloc[:1].to_dict()


class ExtratorDeEstudante(BaseModel):
    estudante: str = Field(
        description="Nome do estudante informado. Exemplo: joão, carlos, maria, fátima."
    )


class DadosDeEstudante(BaseTool):
    name = "DadosDeEstudante"
    description = "Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico."

    def _run(self, input: str) -> str:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
        format_instructions = parser.get_format_instructions()

        template = PromptTemplate(
            template="""Você deve analisar o input: {input}
                         e extrair o nome do estudante informado, . 
                         A saída deve estar no formato JSON: {format_instructions}""",
            input_variables=["input"],
            partial_variables={"format_instructions": format_instructions},
        )

        chain = template | llm | parser

        try:
            result = chain.invoke({"input": input})
            estudante = result["estudante"].lower()

            dados = busca_dados_estudante(estudante)

            return json.dumps(dados)
        except Exception as e:
            return f"Erro ao processar o input: {str(e)}"
