from langchain.agents import AgentExecutor
from agent_gemini import AgenteGemini
from dotenv import load_dotenv

load_dotenv()


pergunta = "Quais sao os dados da Ana e da Bianca?"
# pergunta = "Quais sao os dados da Bianca?"

agente = AgenteGemini()
agent_executor = AgentExecutor(agent=agente.agente, tools=agente.tools, verbose=True)
agent_executor.invoke({"input": pergunta})