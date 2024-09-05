from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from estudante import DadosDeEstudante
import os
from dotenv import load_dotenv

load_dotenv()

class AgenteGemini:
    def __init__(self):
        dados_de_estudante = DadosDeEstudante()

        self.tools = [
            Tool(
                name = dados_de_estudante.name,
                func = dados_de_estudante._run,
                description = dados_de_estudante.description 
            )
        ]

        llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    api_key=os.getenv("GOOGLE_API_KEY")
                    
                )

        # llm = ChatOpenAI(
        #     model="gpt-4o",
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        # )

        prompt = hub.pull("hwchase17/openai-functions-agent")

        self.agente = create_tool_calling_agent(llm, self.tools, prompt)

        # self.agente = create_openai_tools_agent(llm, self.tools, prompt)