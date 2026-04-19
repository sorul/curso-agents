from pydantic import BaseModel, Field
from agents import Agent

HOW_MANY_SEARCHES = 3

INSTRUCTIONS = f"""Eres un asistente de investigación útil. Recibirás un brief con una consulta inicial
y aclaraciones del usuario. Produce un conjunto de búsquedas web para responder la consulta respetando
el objetivo, alcance, restricciones y preferencias indicadas.

Salida: {HOW_MANY_SEARCHES} términos para consultar."""


class WebSearchItem(BaseModel):
    reason: str = Field(description="Tu razonamiento de por qué esta búsqueda es importante para la consulta.")
    query: str = Field(description="El término de búsqueda a usar para la búsqueda web.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="Una lista de búsquedas web a realizar para responder la consulta.")
    
planner_agent = Agent(
    name="Agente de planificación",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan,
)
