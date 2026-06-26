from typing import Optional, Tuple


def extract_thinking_content(content: str) -> Tuple[Optional[str], str]:
  """Extract thinking content from response text between <think> tags."""
  if not content or "</think>" not in content:
    return None, content

  # Find the end of thinking content
  end_idx = content.find("</think>")

  # Look for opening <think> tag, if not found, assume thinking starts at beginning
  start_idx = content.find("<think>")
  if start_idx == -1:
    reasoning_content = content[:end_idx].strip()
  else:
    start_idx = start_idx + len("<think>")
    reasoning_content = content[start_idx:end_idx].strip()

  output_content = content[end_idx + len("</think>") :].strip()

  return reasoning_content, output_content
