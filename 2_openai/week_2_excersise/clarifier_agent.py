from pydantic import BaseModel, Field
from agents import Agent

HOW_MANY_QUESTIONS = 3

INSTRUCTIONS = f"""
Eres un analista de investigación. Tu tarea es leer una consulta inicial y generar preguntas de aclaración
que ayuden a preparar una investigación profunda más útil.

Genera entre 1 y {HOW_MANY_QUESTIONS} preguntas. Pregunta solo lo que realmente mejore la investigación:
objetivo, audiencia, alcance geográfico o sectorial, periodo temporal, profundidad, restricciones,
comparaciones deseadas o aspectos que deben quedar fuera.

No respondas la consulta ni empieces la investigación. Las preguntas deben ser claras, concretas y fáciles
de responder por una persona no técnica.
"""


class ClarifyingQuestion(BaseModel):
    question: str = Field(description="Pregunta clara para afinar la investigación.")
    reason: str = Field(description="Por qué esta pregunta ayuda a mejorar la búsqueda profunda.")


class ClarifyingQuestions(BaseModel):
    questions: list[ClarifyingQuestion] = Field(
        description="Lista de preguntas de aclaración para el usuario."
    )


clarifier_agent = Agent(
    name="Agente de aclaración",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ClarifyingQuestions,
)
