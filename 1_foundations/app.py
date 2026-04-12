from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAI, OpenAIError
from enum import Enum
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from pypdf import PdfReader
import psutil
import gradio as gr

OPENAI_MODEL = 'gpt-4o-mini'
OLLAMA_MODEL = 'llama3.2:latest'
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
DB_PATH = Path("me/conversations.db")
SERVER_PORT = 7863

load_dotenv(override=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class Provider(str, Enum):
  """Proveedores de modelos disponibles en la interfaz."""

  OPENAI = "OpenAI"
  OLLAMA = "Ollama"


def init_database():
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  with sqlite3.connect(DB_PATH) as connection:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS contacts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          email TEXT NOT NULL,
          name TEXT,
          notes TEXT,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS unknown_questions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          question TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def kill_processes_using_port(port):
  current_pid = psutil.Process().pid
  killed_processes = []

  for connection in psutil.net_connections(kind="inet"):
    if connection.status != psutil.CONN_LISTEN:
      continue
    if not connection.laddr or connection.laddr.port != port:
      continue
    if not connection.pid or connection.pid == current_pid:
      continue

    process = psutil.Process(connection.pid)
    logger.info(
        "Killing process using port %s: pid=%s name=%s",
        port,
        process.pid,
        process.name(),
    )
    process.terminate()
    killed_processes.append(process)

  _, alive = psutil.wait_procs(killed_processes, timeout=3)
  for process in alive:
    logger.warning("Process pid=%s did not terminate cleanly; killing it", process.pid)
    process.kill()

  if killed_processes:
    time.sleep(0.5)


def record_user_details(
    email, name="Nombre no indicado", notes="no proporcionadas"
):
  with sqlite3.connect(DB_PATH) as connection:
    cursor = connection.execute(
        """
        INSERT INTO contacts (email, name, notes)
        VALUES (?, ?, ?)
        """,
        (email, name, notes),
    )
  logger.info("Contact recorded in SQLite with id=%s", cursor.lastrowid)
  return {"recorded": "ok", "id": cursor.lastrowid}


def record_unknown_question(question):
  with sqlite3.connect(DB_PATH) as connection:
    cursor = connection.execute(
        """
        INSERT INTO unknown_questions (question)
        VALUES (?)
        """,
        (question,),
    )
  logger.info("Unknown question recorded in SQLite with id=%s", cursor.lastrowid)
  return {"recorded": "ok", "id": cursor.lastrowid}


record_user_details_json = {
    "name":
        "record_user_details",
    "description":
        "Utiliza esta herramienta solo cuando el usuario haya proporcionado explícitamente su dirección de correo electrónico en la conversación. Nunca inventes ni supongas direcciones de email.",
    "parameters":
        {
            "type": "object",
            "properties":
                {
                    "email":
                        {
                            "type": "string",
                            "description": "La dirección de email del usuario"
                        },
                    "name":
                        {
                            "type": "string",
                            "description": "El nombre del usuario, si se indica"
                        },
                    "notes":
                        {
                            "type":
                                "string",
                            "description":
                                "¿Alguna información adicional sobre la conversación que valga la pena registrar para dar contexto?"
                        }
                },
            "required": ["email"],
            "additionalProperties": False
        }
}

record_unknown_question_json = {
    "name":
        "record_unknown_question",
    "description":
        "Utiliza siempre esta herramienta para registrar cualquier pregunta que no haya podido responder porque no se sabía la respuesta.",
    "parameters":
        {
            "type": "object",
            "properties":
                {
                    "question":
                        {
                            "type": "string",
                            "description": "La pregunta no sabe responderse"
                        },
                },
            "required": ["question"],
            "additionalProperties": False
        }
}

tools = [
    {
        "type": "function",
        "function": record_user_details_json
    }, {
        "type": "function",
        "function": record_unknown_question_json
    }
]

init_database()


class Me:
  def __init__(self):
    self.openai = OpenAI()
    # Ollama expone una API local compatible con OpenAI en http://localhost:11434/v1.
    # Podemos reutilizar el cliente oficial de OpenAI apuntando su base_url a Ollama
    # en lugar de a los servidores de OpenAI. El cliente exige un valor api_key,
    # pero Ollama local no lo valida, asi que "ollama" es solo un placeholder.
    self.ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    self.name = "Cayetano Romero"
    reader = PdfReader("me/linkedin.pdf")
    self.linkedin = ""
    for page in reader.pages:
      text = page.extract_text()
      if text:
        self.linkedin += text
    with open("me/summary.txt", "r", encoding="utf-8") as f:
      self.summary = f.read()

  def _email_was_provided_by_user(self, email, messages):
    if not email:
      return False

    for message in messages:
      if hasattr(message, "model_dump"):
        message = message.model_dump()

      if message.get("role") != "user":
        continue

      content = message.get("content") or ""
      if not isinstance(content, str):
        continue

      emails_in_message = EMAIL_PATTERN.findall(content)
      if email in emails_in_message:
        return True

    return False

  def handle_tool_call(self, tool_calls, messages):
    results = []
    for tool_call in tool_calls:
      tool_name = tool_call.function.name
      arguments = json.loads(tool_call.function.arguments)
      logger.info("Tool called: %s", tool_name)
      tool = globals().get(tool_name)
      email_provided = self._email_was_provided_by_user(
          email=arguments.get("email"),
          messages=messages,
      )
      if tool_name == "record_user_details" and not email_provided:
        logger.warning(
            "Rejected record_user_details because the email was not provided by the user: %s",
            arguments.get("email")
        )
        result = {
            "recorded": "rejected",
            "reason": "email_not_provided_by_user"
        }
      else:
        result = tool(**arguments) if tool else {}
      results.append(
          {
              "role": "tool",
              "content": json.dumps(result),
              "tool_call_id": tool_call.id
          }
      )
    return results

  def system_prompt(self):
    system_prompt = f"""Actúas como {self.name}. Respondes preguntas en el sitio web de {self.name}, en particular preguntas relacionadas con la trayectoria profesional, los antecedentes, las habilidades y la experiencia de {self.name}.
            Tu responsabilidad es representar a {self.name} en las interacciones del sitio web con la mayor fidelidad posible.
            Se te proporciona un resumen de la trayectoria profesional y el perfil de LinkedIn de {self.name} que puedes usar para responder preguntas.
            Muestra un tono profesional y atractivo, como si hablaras con un cliente potencial o un futuro empleador que haya visitado el sitio web.
            Si no sabes la respuesta a alguna pregunta, usa la herramienta 'record_unknown_question' para registrar la pregunta que no pudiste responder, incluso si se trata de algo trivial o no relacionado con tu trayectoria profesional.
            Si el usuario participa en una conversación, intenta que se ponga en contacto por correo electrónico; pídele su correo electrónico.
            Usa la herramienta 'record_user_details' solamente si el usuario ha escrito explícitamente su email en la conversación.
            Nunca inventes, supongas ni completes direcciones de correo electrónico."""

    system_prompt += f"\n\n## Resumen:\n{self.summary}\n\n## Perfil de LinkedIn:\n{self.linkedin}\n\n"
    system_prompt += f"En este contexto, por favor chatea con el usuario, manteniéndote siempre en el personaje de {self.name}."
    return system_prompt

  def chat(self, message, history, provider=Provider.OLLAMA.value):
    provider = provider or Provider.OLLAMA.value
    client = self.ollama if provider == Provider.OLLAMA.value else self.openai
    model = OLLAMA_MODEL if provider == Provider.OLLAMA.value else OPENAI_MODEL
    messages = [{"role": "system", "content": self.system_prompt()}]
    for item in history:
      if isinstance(item, dict):
        messages.append({"role": item["role"], "content": item["content"]})
      else:
        user_message, assistant_message = item
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": message})
    done = False
    while not done:
      try:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools
        )
      except AuthenticationError:
        return "La clave de OpenAI configurada no es válida. Selecciona Ollama en el desplegable o actualiza OPENAI_API_KEY en el archivo .env."
      except APIConnectionError:
        return "No puedo conectar con Ollama. Comprueba que Ollama esté arrancado y que el modelo llama3.2:latest esté disponible."
      except OpenAIError as error:
        return f"El proveedor {provider} devolvió un error: {error}"
      if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_calls = message.tool_calls
        results = self.handle_tool_call(tool_calls, messages)
        # Anadimos el mensaje del assistant donde pide ejecutar una o varias herramientas.
        messages.append(message)
        # Anadimos los resultados (1 por herramienta llamada) devueltos por esas herramientas como mensajes role="tool".
        messages.extend(results)
      else:
        done = True
    reply = response.choices[0].message.content
    logger.info("%s: %s", model, reply)
    return reply


# Interfaz en dos columnas: chat a la izquierda, prompt capturado a la derecha.
with gr.Blocks() as interface:
  me = Me()
  with gr.Row():
    with gr.Column(scale=3):
      gr.ChatInterface(
          fn=me.chat,
          additional_inputs=[
              gr.Dropdown(
                  choices=[provider.value for provider in Provider],
                  value=Provider.OLLAMA.value,
                  label="Provider",
              )
          ],
      )

# if __name__ == "__main__":
#   me = Me()
#   gr.ChatInterface(me.chat, type="messages").launch()

# - declaro una variable global "interfaz"
# - en el main cargo el .env y lanzo la interfaz
# - lanzo la interfaz en el puerto 7863
# - en el terminal ejecuto "gradio week5/week5_exercise.py"
# - los cambios en el html se actualizan automáticamente
if __name__ == "__main__":
  kill_processes_using_port(SERVER_PORT)
  interface.launch(server_name="0.0.0.0", server_port=SERVER_PORT, inbrowser=False)
