import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List

import sendgrid
from agents import Agent, ModelSettings, Runner, function_tool, trace
from dotenv import load_dotenv
from sendgrid.helpers.mail import Content, Email, Mail, To


"""
Pipeline de SDR automatizado con OpenAI Agents.

Secuencia del flujo:
1. Se crean tres agentes de ventas con estilos distintos.
2. Los tres agentes generan candidatos de email en paralelo.
3. Un manager recibe los tres candidatos y elige el mejor.
4. Un agente de email crea el asunto, convierte el cuerpo a HTML y lo envía.

La orquestación importante se hace en Python para controlar qué partes se
paralelizan. Solo se paraleliza la generación de candidatos, porque son tareas
independientes. La selección y el envío se hacen en orden para evitar múltiples
handoffs o envíos duplicados.
"""


MODEL = "gpt-4o-mini"
FROM_EMAIL = "cromerovargas2d@gmail.com"
TO_EMAIL = "cromerovargas2d@gmail.com"
MESSAGE = "Envía un correo electrónico de ventas en frío dirigido a 'Estimado director ejecutivo'"


@dataclass(frozen=True)
class SalesCandidate:
    """Email candidato generado por uno de los agentes de ventas."""

    agent_name: str
    email_body: str


@dataclass(frozen=True)
class SDRPipeline:
    """Agrupa los agentes que participan en el flujo completo."""

    sales_agents: List[Agent]
    sales_manager: Agent
    emailer_agent: Agent


@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """
    Tool usada por el Email Manager para enviar el correo final.

    El decorador @function_tool permite que un Agent invoque esta función como
    herramienta. La función espera recibir ya el asunto y el HTML definitivos.
    """

    api_key = os.environ.get("SENDGRID_API_KEY")
    if not api_key:
        raise ValueError("SENDGRID_API_KEY no está configurada")

    sg = sendgrid.SendGridAPIClient(api_key=api_key)
    content = Content("text/html", html_body)
    mail = Mail(Email(FROM_EMAIL), To(TO_EMAIL), subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)  # type: ignore
    return {"status": "success", "code": str(response.status_code)}


def create_subject_writer() -> Agent:
    """Crea el agente especializado en generar asuntos de email."""

    return Agent(
        name="Escritor de asunto de correo electrónico",
        instructions=(
            "Puedes escribir un asunto para un correo electrónico de ventas en frío. "
            "Se te proporciona un mensaje y necesitas escribir un asunto para un "
            "correo electrónico que probablemente obtenga respuesta."
        ),
        model=MODEL,
    )


def create_html_converter() -> Agent:
    """Crea el agente especializado en convertir texto o markdown a HTML."""

    return Agent(
        name="Conversor de cuerpo de correo electrónico HTML",
        instructions=(
            "Puedes convertir un cuerpo de correo electrónico de texto a HTML. "
            "Se te proporciona un cuerpo de correo electrónico de texto que puede "
            "tener algo de markdown y necesitas convertirlo a un HTML simple, "
            "claro y atractivo."
        ),
        model=MODEL,
    )


def create_emailer_agent() -> Agent:
    """
    Crea el agente responsable de preparar y enviar el email elegido.

    Este agente tiene tres tools:
    - subject_writer: genera el asunto.
    - html_converter: convierte el cuerpo a HTML.
    - send_html_email: envía el email real por SendGrid.

    Aquí desactivamos parallel_tool_calls para forzar una secuencia segura:
    asunto -> HTML -> envío.
    """

    subject_tool = create_subject_writer().as_tool(
        tool_name="subject_writer",
        tool_description="Escribe un asunto para un correo electrónico de ventas en frío",
    )
    html_tool = create_html_converter().as_tool(
        tool_name="html_converter",
        tool_description="Convierte un cuerpo de correo electrónico de texto a HTML",
    )

    return Agent(
        name="Email Manager",
        instructions=(
            "Eres un formateador y remitente de correos electrónicos. "
            "Recibes el cuerpo definitivo de un correo electrónico para enviarlo. "
            "Primero usas subject_writer para escribir un asunto. "
            "Luego usas html_converter para convertir el cuerpo a HTML. "
            "Finalmente usas send_html_email para enviar el correo con el asunto "
            "y el cuerpo HTML. Ejecuta estos pasos en orden y una sola vez."
        ),
        tools=[subject_tool, html_tool, send_html_email],
        model_settings=ModelSettings(parallel_tool_calls=False),
        model=MODEL,
    )


def create_sales_agents() -> List[Agent]:
    """
    Crea los tres agentes que proponen emails en estilos distintos.

    Estos agentes no envían nada. Solo generan propuestas de cuerpo de email.
    Por eso se pueden ejecutar en paralelo sin riesgo de duplicar envíos.
    """

    agent_configs = [
        (
            "Agente de ventas profesional",
            "Eres un agente de ventas que trabaja para ComplAI, una empresa que "
            "ofrece una herramienta SaaS para garantizar el cumplimiento de SOC2 "
            "y prepararse para auditorías, impulsada por IA. Redactas correos "
            "electrónicos en frío profesionales y serios.",
        ),
        (
            "Agente de ventas atractivo",
            "Eres un agente de ventas con sentido del humor y atractivo que "
            "trabaja para ComplAI, una empresa que ofrece una herramienta SaaS "
            "para garantizar el cumplimiento de SOC2 y prepararse para auditorías, "
            "impulsada por IA. Redactas correos electrónicos en frío ingeniosos "
            "y atractivos que probablemente obtengan respuesta.",
        ),
        (
            "Agente de ventas ocupado",
            "Eres un agente de ventas muy activo que trabaja para ComplAI, una "
            "empresa que ofrece una herramienta SaaS para garantizar el cumplimiento "
            "de SOC2 y prepararse para auditorías, impulsada por IA. Redactas "
            "correos electrónicos en frío concisos y directos.",
        ),
    ]

    return [
        Agent(name=name, instructions=instructions, model=MODEL)
        for name, instructions in agent_configs
    ]


def create_sales_manager() -> Agent:
    """
    Crea el agente que selecciona el mejor candidato.

    Este manager no tiene tools ni handoffs. Su única responsabilidad es
    decidir cuál de los tres textos es mejor y devolver el cuerpo definitivo.
    """

    return Agent(
        name="Manager de ventas",
        instructions=(
            "Eres un gerente de ventas que trabaja para ComplAI. Recibes varios "
            "correos electrónicos de ventas en frío candidatos y eliges el más "
            "efectivo. Evalúas claridad, probabilidad de respuesta, adecuación "
            "al destinatario y concisión. Devuelve únicamente el cuerpo completo "
            "del correo elegido, sin explicación ni metacomentarios."
        ),
        model=MODEL,
    )


def create_pipeline() -> SDRPipeline:
    """Construye todos los agentes del pipeline en un único objeto."""

    return SDRPipeline(
        sales_agents=create_sales_agents(),
        sales_manager=create_sales_manager(),
        emailer_agent=create_emailer_agent(),
    )


async def generate_candidate(agent: Agent, message: str) -> SalesCandidate:
    """Ejecuta un agente de ventas y empaqueta su salida como candidato."""

    result = await Runner.run(agent, message)
    return SalesCandidate(agent_name=agent.name, email_body=str(result.final_output))


async def generate_candidates(agents: List[Agent], message: str) -> List[SalesCandidate]:
    """
    Genera todos los candidatos en paralelo.

    Esta es la única parte paralelizada del flujo. Cada agente trabaja con el
    mismo briefing y produce una propuesta independiente.
    """

    tasks = [generate_candidate(agent, message) for agent in agents]
    return await asyncio.gather(*tasks)


def build_selection_prompt(message: str, candidates: List[SalesCandidate]) -> str:
    """Construye el prompt que recibirá el manager para elegir el mejor email."""

    formatted_candidates = "\n\n".join(
        f"## {candidate.agent_name}\n{candidate.email_body}"
        for candidate in candidates
    )
    return (
        f"Petición original:\n{message}\n\n"
        f"Candidatos generados:\n{formatted_candidates}\n\n"
        "Elige el mejor correo y devuelve únicamente el cuerpo del correo elegido."
    )


async def select_best_email(
    sales_manager: Agent,
    message: str,
    candidates: List[SalesCandidate],
) -> str:
    """Pide al manager que elija uno de los candidatos generados."""

    prompt = build_selection_prompt(message, candidates)
    result = await Runner.run(sales_manager, prompt)
    return str(result.final_output)


async def send_selected_email(emailer_agent: Agent, email_body: str):
    """
    Pide al Email Manager que prepare y envíe el email elegido.

    Llegados a este punto ya hay un único cuerpo de email. El envío se mantiene
    separado de la selección para que la operación con efectos externos ocurra
    una sola vez.
    """

    prompt = (
        "Envía el siguiente correo electrónico de ventas en frío. "
        "No lo reescribas; solo crea un asunto, conviértelo a HTML y envíalo.\n\n"
        f"{email_body}"
    )
    return await Runner.run(emailer_agent, prompt)


async def run_pipeline(message: str = MESSAGE):
    """
    Ejecuta el flujo completo.

    Orden real de ejecución:
    1. Carga variables de entorno.
    2. Crea los agentes.
    3. Genera tres candidatos en paralelo.
    4. Selecciona el mejor candidato.
    5. Envía el candidato seleccionado.
    """

    load_dotenv(override=True)
    pipeline = create_pipeline()

    with trace("Automated SDR"):
        # Paralelo: tres propuestas independientes.
        candidates = await generate_candidates(pipeline.sales_agents, message)

        # Secuencial: una única decisión sobre los candidatos.
        selected_email = await select_best_email(
            pipeline.sales_manager,
            message,
            candidates,
        )

        # Secuencial: una única operación externa de envío.
        return await send_selected_email(pipeline.emailer_agent, selected_email)


async def main():
    """Punto de entrada async para ejecutar el script desde terminal."""

    result = await run_pipeline()
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
