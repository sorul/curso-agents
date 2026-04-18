from agents import Agent, ModelSettings, function_tool

INSTRUCTIONS = (
    "Eres un asistente de investigación. Dado un término de búsqueda, usa la herramienta mock_web_search "
    "para obtener resultados simulados sin coste de búsqueda web y produce un resumen conciso de esos resultados. "
    "El resumen debe tener 2-3 párrafos y menos de 300 palabras. Captura los puntos principales. Escribe de "
    "manera concisa, no es necesario tener frases completas o buena gramática. Esto será consumido por alguien "
    "que sintetiza un informe, por lo que es vital que captures la esencia y ignores cualquier fluff. No "
    "incluyas ningún comentario adicional más que el resumen en sí."
)


@function_tool
def mock_web_search(query: str) -> str:
    """Devuelve resultados de búsqueda simulados para evitar usar WebSearchTool."""
    return (
        f"Resultados simulados para: {query}\n\n"
        "1. Fuente simulada A: resumen general del tema, principales conceptos y contexto relevante.\n"
        "2. Fuente simulada B: posibles tendencias, oportunidades, riesgos y puntos de debate habituales.\n"
        "3. Fuente simulada C: ejemplos prácticos, actores relacionados y preguntas que conviene investigar.\n\n"
        "Nota: estos datos son mock y no provienen de una búsqueda real en internet."
    )


search_agent = Agent(
    name="Agente de búsqueda",
    instructions=INSTRUCTIONS,
    tools=[mock_web_search],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)
