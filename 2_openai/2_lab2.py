import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List

import sendgrid
from agents import Agent, ModelSettings, Runner, function_tool, trace
from dotenv import load_dotenv
from sendgrid.helpers.mail import Content, Email, Mail, To


MODEL = "gpt-4o-mini"
FROM_EMAIL = "cromerovargas2d@gmail.com"
TO_EMAIL = "cromerovargas2d@gmail.com"
MESSAGE = "Envía un correo electrónico de ventas en frío dirigido a 'Estimado director ejecutivo'"


@dataclass(frozen=True)
class SalesCandidate:
    agent_name: str
    email_body: str


@dataclass(frozen=True)
class SDRPipeline:
    sales_agents: List[Agent]
    sales_manager: Agent
    emailer_agent: Agent


@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """Envía un correo electrónico con el asunto y el cuerpo HTML."""
    api_key = os.environ.get("SENDGRID_API_KEY")
    if not api_key:
        raise ValueError("SENDGRID_API_KEY no está configurada")

    sg = sendgrid.SendGridAPIClient(api_key=api_key)
    content = Content("text/html", html_body)
    mail = Mail(Email(FROM_EMAIL), To(TO_EMAIL), subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)  # type: ignore
    return {"status": "success", "code": str(response.status_code)}


def create_subject_writer() -> Agent:
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
    return SDRPipeline(
        sales_agents=create_sales_agents(),
        sales_manager=create_sales_manager(),
        emailer_agent=create_emailer_agent(),
    )


async def generate_candidate(agent: Agent, message: str) -> SalesCandidate:
    result = await Runner.run(agent, message)
    return SalesCandidate(agent_name=agent.name, email_body=str(result.final_output))


async def generate_candidates(agents: List[Agent], message: str) -> List[SalesCandidate]:
    tasks = [generate_candidate(agent, message) for agent in agents]
    return await asyncio.gather(*tasks)


def build_selection_prompt(message: str, candidates: List[SalesCandidate]) -> str:
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
    prompt = build_selection_prompt(message, candidates)
    result = await Runner.run(sales_manager, prompt)
    return str(result.final_output)


async def send_selected_email(emailer_agent: Agent, email_body: str):
    prompt = (
        "Envía el siguiente correo electrónico de ventas en frío. "
        "No lo reescribas; solo crea un asunto, conviértelo a HTML y envíalo.\n\n"
        f"{email_body}"
    )
    return await Runner.run(emailer_agent, prompt)


async def run_pipeline(message: str = MESSAGE):
    load_dotenv(override=True)
    pipeline = create_pipeline()

    with trace("Automated SDR"):
        candidates = await generate_candidates(pipeline.sales_agents, message)
        selected_email = await select_best_email(
            pipeline.sales_manager,
            message,
            candidates,
        )
        return await send_selected_email(pipeline.emailer_agent, selected_email)


async def main():
    result = await run_pipeline()
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
