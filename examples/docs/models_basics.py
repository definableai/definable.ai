from definable.model import Claude, OpenAIChat, resolve_model_string


openai = OpenAIChat(id="gpt-4o-mini")
claude = Claude(id="claude-sonnet-4-5-20250929", api_key="test-key")
resolved = resolve_model_string("openai/gpt-4o-mini")

assert openai.id == "gpt-4o-mini"
assert claude.id.startswith("claude")
assert type(resolved).__name__ == "OpenAIChat"
