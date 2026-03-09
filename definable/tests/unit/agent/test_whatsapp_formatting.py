"""Tests for Markdown → WhatsApp formatting conversion."""

from definable.agent.interface.whatsapp.formatting import markdown_to_whatsapp


class TestMarkdownToWhatsApp:
  def test_plain_text_passthrough(self):
    assert markdown_to_whatsapp("Hello world") == "Hello world"

  def test_empty(self):
    assert markdown_to_whatsapp("") == ""

  def test_bold_double_asterisk(self):
    assert markdown_to_whatsapp("This is **bold** text") == "This is *bold* text"

  def test_bold_double_underscore(self):
    assert markdown_to_whatsapp("This is __bold__ text") == "This is *bold* text"

  def test_strikethrough(self):
    assert markdown_to_whatsapp("This is ~~deleted~~ text") == "This is ~deleted~ text"

  def test_inline_code(self):
    result = markdown_to_whatsapp("Use `pip install` to install")
    assert "```pip install```" in result

  def test_fenced_code_block(self):
    md = "```python\nprint('hello')\n```"
    result = markdown_to_whatsapp(md)
    assert "```\nprint('hello')\n```" in result

  def test_heading_h1(self):
    assert markdown_to_whatsapp("# Title") == "*Title*"

  def test_heading_h3(self):
    assert markdown_to_whatsapp("### Subtitle") == "*Subtitle*"

  def test_link(self):
    result = markdown_to_whatsapp("[Click here](https://example.com)")
    assert result == "Click here (https://example.com)"

  def test_image(self):
    result = markdown_to_whatsapp("![Alt text](https://example.com/img.png)")
    assert result == "Alt text: https://example.com/img.png"

  def test_horizontal_rule(self):
    assert markdown_to_whatsapp("---") == "───"
    assert markdown_to_whatsapp("***") == "───"

  def test_bullet_list_preserved(self):
    md = "- Item 1\n- Item 2"
    assert markdown_to_whatsapp(md) == "- Item 1\n- Item 2"

  def test_numbered_list_preserved(self):
    md = "1. First\n2. Second"
    assert markdown_to_whatsapp(md) == "1. First\n2. Second"

  def test_blockquote_preserved(self):
    md = "> This is a quote"
    assert markdown_to_whatsapp(md) == "> This is a quote"

  def test_mixed_formatting(self):
    md = "# Welcome\n\nThis is **bold** and ~~old~~ text.\n\n```\ncode\n```"
    result = markdown_to_whatsapp(md)
    assert "*Welcome*" in result
    assert "*bold*" in result
    assert "~old~" in result
    assert "```\ncode\n```" in result

  def test_code_block_not_mangled(self):
    """Code blocks should not have internal formatting applied."""
    md = "```\n**not bold** ~~not strike~~\n```"
    result = markdown_to_whatsapp(md)
    assert "**not bold**" in result
    assert "~~not strike~~" in result

  def test_none_input(self):
    # Passing None should be handled gracefully
    assert markdown_to_whatsapp(None) is None  # type: ignore[arg-type]


class TestMarkdownEdgeCases:
  def test_multiple_bold_in_one_line(self):
    result = markdown_to_whatsapp("**A** and **B**")
    assert result == "*A* and *B*"

  def test_nested_formatting_best_effort(self):
    # WhatsApp doesn't support nested formatting well,
    # but we shouldn't crash
    result = markdown_to_whatsapp("**bold and ~~strike~~**")
    assert "~strike~" in result

  def test_url_without_link_syntax(self):
    text = "Visit https://example.com for info"
    assert markdown_to_whatsapp(text) == text

  def test_multiple_code_blocks(self):
    md = "```\nblock1\n```\ntext\n```\nblock2\n```"
    result = markdown_to_whatsapp(md)
    assert "block1" in result
    assert "block2" in result
