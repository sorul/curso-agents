import json


def messages_to_readable_conversation(messages):
  """Convierte una lista de mensajes estilo OpenAI en una conversacion legible."""
  lines = []

  for message in messages:
    if hasattr(message, "model_dump"):
      message = message.model_dump()

    role = message.get("role", "unknown")
    content = message.get("content")

    if content and role != "tool":
      lines.append(f"{role}: {content}")

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
      if hasattr(tool_call, "model_dump"):
        tool_call = tool_call.model_dump()

      function = tool_call.get("function", {})
      name = function.get("name", "unknown_tool")
      arguments = _format_json(function.get("arguments", "{}"))
      lines.append(f"{role} -> tool {name}: {arguments}")

    if role == "tool":
      tool_call_id = message.get("tool_call_id", "unknown_call")
      lines.append(f"tool result ({tool_call_id}): {_format_json(content)}")

  return "\n".join(lines)


def _format_json(value):
  if not isinstance(value, str):
    return json.dumps(value, ensure_ascii=False)

  try:
    parsed = json.loads(value)
  except json.JSONDecodeError:
    return value

  return json.dumps(parsed, ensure_ascii=False)
