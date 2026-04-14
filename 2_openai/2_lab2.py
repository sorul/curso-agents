from dotenv import load_dotenv
from agents import Agent, Runner, trace, function_tool, ModelSettings
from typing import Dict
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content
import asyncio

load_dotenv(override=True)


@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
  """ Envía un correo electrónico con el asunto y el cuerpo HTML a todos los clientes potenciales de ventas """
  sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
  from_email = Email("cromerovargas2d@gmail.com")
  to_email = To("cromerovargas2d@gmail.com")
  content = Content("text/html", html_body)
  mail = Mail(from_email, to_email, subject, content).get()
  response = sg.client.mail.send.post(request_body=mail) # type: ignore
  return {"status": "success", "code": str(response.status_code)}


# Subject Agent
subject_instructions = "Puedes escribir un asunto para un correo electrónico de ventas en frío. \
    Se te proporciona un mensaje y necesitas escribir un asunto para un correo electrónico que probablemente obtenga respuesta."

subject_writer = Agent(
    name="Escritor de asunto de correo electrónico",
    instructions=subject_instructions,
    model="gpt-4o-mini"
)
subject_tool = subject_writer.as_tool(
    tool_name="subject_writer",
    tool_description=
    "Escribe un asunto para un correo electrónico de ventas en frío"
)

# HTML Converter Agent
html_instructions = "Puedes convertir un cuerpo de correo electrónico de texto a un cuerpo de correo electrónico HTML. \
    Se te proporciona un cuerpo de correo electrónico de texto que puede tener algún markdown \
    y necesitas convertirlo a un cuerpo de correo electrónico HTML con un diseño simple, claro y atractivo."

html_converter = Agent(
    name="Conversor de cuerpo de correo electrónico HTML",
    instructions=html_instructions,
    model="gpt-4o-mini"
)
html_tool = html_converter.as_tool(
    tool_name="html_converter",
    tool_description=
    "Convierte un cuerpo de correo electrónico de texto a un cuerpo de correo electrónico HTML"
)

# Email Manager Agent
emailer_agent_tools = [subject_tool, html_tool, send_html_email]

emailer_agent_instructions = "Eres un formateador y remitente de correos electrónicos. \
    Recibes el cuerpo de un correo electrónico para enviarlo. \
    Primero usas la herramienta subject_writer para escribir un asunto para el correo electrónico, \
    luego usas la herramienta html_converter para convertir el cuerpo a HTML. \
    Finalmente, usas la herramienta send_html_email para enviar el correo electrónico con el asunto y el cuerpo HTML."

emailer_agent = Agent(
    name="Email Manager",
    instructions=emailer_agent_instructions,
    tools=emailer_agent_tools,
    model="gpt-4o-mini",
    handoff_description="Convierte un email a HTML y lo envía"
)

###########################################################################

instructions1 = "Eres un agente de ventas que trabaja para ComplAI, \
    una empresa que ofrece una herramienta SaaS para garantizar el cumplimiento de SOC2\
    y prepararse para auditorías, impulsada por IA. Redactas correos electrónicos en frío profesionales y serios."

instructions2 = "Eres un agente de ventas con sentido del humor y atractivo \
    que trabaja para ComplAI, una empresa que ofrece una herramienta SaaS para \
    garantizar el cumplimiento de SOC2 y prepararse para auditorías, impulsada por IA. \
    Redactas correos electrónicos en frío ingeniosos y atractivos que probablemente obtengan respuesta."

instructions3 = "Eres un agente de ventas muy activo que trabaja para ComplAI, \
    una empresa que ofrece una herramienta SaaS para garantizar el cumplimiento de SOC2\
    y prepararse para auditorías, impulsada por IA. Redactas correos electrónicos en frío concisos y directos."

sales_agent1 = Agent(
        name="Agente de ventas profesional",
        instructions=instructions1,
        model="gpt-4o-mini"
)

sales_agent2 = Agent(
        name="Agente de ventas atractivo",
        instructions=instructions2,
        model="gpt-4o-mini"
)

sales_agent3 = Agent(
        name="Agente de ventas ocpado",
        instructions=instructions3,
        model="gpt-4o-mini"
)


description = "Escribe un correo electrónico de ventas en frío"
tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

sales_manager_tools = [tool1, tool2, tool3]
handoffs = [emailer_agent]


sales_manager_instructions = "Eres un gerente de ventas que trabaja para ComplAI. Utilizas las herramientas que se te proporcionan para generar correos electrónicos de ventas en frío. \
Nunca generas correos electrónicos de ventas tú mismo; siempre utilizas las herramientas. \
Pruebas las 3 herramientas del agente de ventas al menos una vez antes de elegir la mejor. \
Puedes usar las herramientas múltiples veces si no estás satisfecho con los resultados del primer intento. \
Seleccionas el mejor correo electrónico usando tu propio criterio sobre cuál será más efectivo. \
Después de elegir el correo electrónico, transfieres al agente Email Manager una sola vez para formatear y enviar el correo. \
No solicites más de un handoff en la misma respuesta."


sales_manager = Agent(
    name="Manager de ventas",
    instructions=sales_manager_instructions,
    tools=sales_manager_tools,
    handoffs=handoffs,
    model_settings=ModelSettings(parallel_tool_calls=False),
    model="gpt-4o-mini")

async def main():
  message = "Envía un correo electrónico de ventas en frío dirigido a 'Estimado director ejecutivo'"

  with trace("Automated SDR"):
    result = await Runner.run(sales_manager, message)

  return result


if __name__ == "__main__":
  asyncio.run(main())
