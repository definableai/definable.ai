"""CLI interface configuration."""

from dataclasses import dataclass

from definable.agent.interface.config import InterfaceConfig


@dataclass(frozen=True)
class CLIConfig(InterfaceConfig):
  """Configuration for the CLI interface.

  Extends InterfaceConfig with CLI-specific display and behavior settings.

  Attributes:
    mode: Display mode — "auto" (detect best), "tui" (Textual), "repl" (Rich).
    prompt: Input prompt string shown to the user.
    show_banner: Whether to display agent info banner on startup.
    show_metrics: Whether to show token counts after runs.
    show_tool_args: Whether to display tool call arguments.
    show_tool_results: Whether to display tool call results.
    show_thinking: Whether to render reasoning/thinking events.
    show_timestamps: Whether to prefix events with timestamps.
    max_content_display: Truncation limit for displayed content.
    markdown_output: Whether to render assistant output as Markdown.
    command_prefix: Prefix character for slash commands.
    enable_completions: Whether to enable slash-command dropdown completions.
    user_id: Default user ID for the CLI session.
    tools_expand: When to auto-expand tool call blocks (TUI mode only).
  """

  platform: str = "cli"
  mode: str = "auto"  # "auto" | "tui" | "repl"
  prompt: str = ">>> "
  show_banner: bool = True
  show_metrics: bool = True
  show_tool_args: bool = True
  show_tool_results: bool = True
  show_thinking: bool = True
  show_timestamps: bool = False
  max_content_display: int = 500
  markdown_output: bool = True
  command_prefix: str = "/"
  enable_completions: bool = True
  user_id: str = "cli-user"
  max_message_length: int = 100_000
  max_concurrent_requests: int = 1
  tools_expand: str = "success"  # "always" | "success" | "fail" | "both" | "never"
