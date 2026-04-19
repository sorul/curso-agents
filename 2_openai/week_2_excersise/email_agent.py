import os
from typing import Dict

import sendgrid
from sendgrid.helpers.mail import Email, Mail, Content, To
from agents import Agent, ModelSettings, function_tool

@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Envía un correo electrónico con el asunto y el cuerpo HTML proporcionados """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("cromerovargas2d@gmail.com")
    to_email = To("cromerovargas2d@gmail.com")
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)
    print("Respuesta de correo electrónico", response.status_code)
    return {"status": "success"}

subject_writer_agent = Agent(
    name="Escritor de asunto",
    instructions=(
        "Recibes un informe de investigación y escribes un asunto breve, claro y "
        "apropiado para enviarlo por correo electrónico. Devuelve solo el asunto."
    ),
    model="gpt-4o-mini",
)

html_writer_agent = Agent(
    name="Escritor HTML",
    instructions=(
        "Recibes un informe en markdown y lo conviertes en un cuerpo HTML limpio, "
        "legible y bien estructurado para correo electrónico. Devuelve solo el HTML."
    ),
    model="gpt-4o-mini",
)

subject_writer_tool = subject_writer_agent.as_tool(
    tool_name="subject_writer",
    tool_description="Escribe un asunto de email para un informe de investigación.",
)

html_writer_tool = html_writer_agent.as_tool(
    tool_name="html_writer",
    tool_description="Convierte un informe markdown en un cuerpo HTML para email.",
)

INSTRUCTIONS = """Puedes enviar un correo electrónico basado en un informe detallado.
Se te proporcionará el informe final en markdown. Debes:
1. Usar subject_writer para crear un asunto apropiado.
2. Usar html_writer para convertir el informe a HTML.
3. Usar send_email para enviar exactamente un correo con ese asunto y ese HTML.

No envíes el correo antes de tener el asunto y el HTML definitivos."""

email_agent = Agent(
    name="Agente de correo electrónico",
    instructions=INSTRUCTIONS,
    tools=[subject_writer_tool, html_writer_tool, send_email],
    model="gpt-4o-mini",
    model_settings=ModelSettings(parallel_tool_calls=False),
)
