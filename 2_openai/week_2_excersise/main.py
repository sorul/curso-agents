import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager

load_dotenv(override=True)


async def generate_questions(query: str):
    if not query.strip():
        return "Escribe primero una consulta inicial."

    questions = await ResearchManager().clarify_query(query)
    formatted_questions = [
        f"{index}. {item.question}"
        for index, item in enumerate(questions.questions, start=1)
    ]
    return "### Preguntas para afinar la investigación\n\n" + "\n\n".join(
        formatted_questions)


async def run(query: str, clarification_answers: str):
    async for chunk in ResearchManager().run(query, clarification_answers):
        yield chunk


with gr.Blocks() as ui:
    gr.Markdown("# Búsqueda Profunda")
    query_textbox = gr.Textbox(label="¿Sobre qué tema te gustaría investigar?")
    generate_questions_button = gr.Button("Generar preguntas",
                                          variant="secondary")
    clarifying_questions = gr.Markdown(label="Preguntas")
    clarification_answers_textbox = gr.Textbox(
        label="Responde a las preguntas anteriores",
        lines=8,
        placeholder="1. ...\n2. ...\n3. ...",
    )
    run_button = gr.Button("Ejecutar investigación", variant="primary")
    report = gr.Markdown(label="Informe")

    generate_questions_button.click(
        fn=generate_questions,
        inputs=query_textbox,
        outputs=clarifying_questions,
    )
    query_textbox.submit(
        fn=generate_questions,
        inputs=query_textbox,
        outputs=clarifying_questions,
    )
    run_button.click(
        fn=run,
        inputs=[query_textbox, clarification_answers_textbox],
        outputs=report,
    )


if __name__ == "__main__":
    ui.launch(
        server_name="0.0.0.0",
        server_port=7863,
        inbrowser=True,
        theme=gr.themes.Default(primary_hue="sky"),
    )
