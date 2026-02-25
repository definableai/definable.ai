# Browser

> Playwright-based browser automation toolkit for Definable AI agents.

The Browser module gives agents full control over a Chrome browser via the Chrome DevTools Protocol (CDP). Agents can navigate pages, read content, fill forms, click buttons, take screenshots, and more — all through 50+ tools exposed as agent-callable functions.

## Quick Start

```python
import asyncio
from definable.agent import Agent
from definable.browser import BrowserToolkit, BrowserConfig

async def main():
    async with BrowserToolkit() as toolkit:
        agent = Agent(
            model="openai/gpt-4o",
            toolkits=[toolkit],
        )
        result = await agent.arun("Go to news.ycombinator.com and list the top 3 stories")
        print(result.content)

asyncio.run(main())
```

> **Requires:** `pip install 'definable[browser]'` then `playwright install chromium`

## Architecture

```
BrowserToolkit (Toolkit subclass)
  │
  ├── PlaywrightBrowser (BaseBrowser implementation)
  │     ├── Playwright CDP connection
  │     ├── Page state tracking (console, errors, network)
  │     ├── Element reference system (e1, e2, ...)
  │     └── Chrome launcher (auto-launch or attach)
  │
  ├── BrowserConfig (frozen dataclass)
  │     ├── Connection: cdp_url, user_data_dir, or fresh launch
  │     ├── Launch: headless, stealth, proxy, no_sandbox
  │     └── Behavior: timeout, viewport, locale
  │
  └── 50+ Tools (Function objects)
        ├── Navigation (4): navigate, back, forward, refresh
        ├── Page state (7): url, title, text, source, attribute, visible, page_info
        ├── Perception (2): snapshot, screenshot
        ├── Interaction (15): click, type, hover, drag, select, fill_form, ...
        ├── Checkboxes (3): check, uncheck, is_checked
        ├── Scrolling (3): scroll_down, scroll_up, scroll_to
        ├── Waiting (4): wait, wait_for_element, wait_for_text, wait_for
        ├── Tabs (4): open, close, list, switch
        ├── Cookies (3): get, set, clear
        ├── Storage (2): get_storage, set_storage
        ├── DOM (2): highlight, remove_elements
        ├── Diagnostics (3): console, errors, network
        ├── Dialogs (1): handle_dialog
        ├── Emulation (1): set_geolocation
        └── PDF (1): print_to_pdf
```

### Module Structure

```
browser/
├── __init__.py              # Public API: BaseBrowser, BrowserConfig, BrowserToolkit
├── base.py                  # BaseBrowser ABC (50+ abstract methods)
├── toolkit.py               # BrowserToolkit (exposes tools to agents)
├── config.py                # BrowserConfig (frozen dataclass)
├── playwright_browser.py    # PlaywrightBrowser implementation
├── chrome_launcher.py       # Chrome process management
├── element_refs.py          # Element reference tracking (e1, e2, ...)
├── page_state.py            # Console, error, network capture
├── url_validator.py         # URL validation
└── events.py                # BrowserActionEvent
```

## API Reference

### BrowserConfig

Immutable configuration. Three connection modes in priority order:

```python
from definable.browser import BrowserConfig

# 1. Fresh Chrome (most common)
config = BrowserConfig(headless=False)

# 2. Persistent profile — stays logged in between runs
config = BrowserConfig(user_data_dir="/tmp/my-profile")

# 3. Attach to your running Chrome
# (launch Chrome with: --remote-debugging-port=9222 --no-first-run)
config = BrowserConfig(cdp_url="ws://127.0.0.1:9222")

# CI / Docker
config = BrowserConfig(headless=True, no_sandbox=True)

# With proxy
config = BrowserConfig(proxy="user:pass@proxy.example.com:8080")
```

**All parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cdp_url` | `str` | `None` | WebSocket URL to attach to running Chrome |
| `user_data_dir` | `str` | `None` | Path to persistent Chrome profile |
| `headless` | `bool` | `False` | Run without visible window |
| `stealth` | `bool` | `True` | Anti-detection measures |
| `no_sandbox` | `bool` | `False` | Disable sandbox (needed in Docker) |
| `proxy` | `str` | `None` | Proxy server (`user:pass@host:port`) |
| `timeout` | `float` | `30.0` | Default timeout in seconds |
| `viewport_width` | `int` | `1280` | Browser viewport width |
| `viewport_height` | `int` | `720` | Browser viewport height |
| `locale` | `str` | `"en-US"` | Browser locale |
| `timezone` | `str` | `None` | Timezone override |
| `user_agent` | `str` | `None` | Custom user agent |
| `executable_path` | `str` | `None` | Path to Chrome binary |
| `extra_args` | `tuple` | `()` | Additional Chrome launch arguments |
| `slow_mo` | `float` | `0.0` | Delay between actions (debugging) |
| `downloads_dir` | `str` | `None` | Download directory |

### BrowserToolkit

The agent-facing toolkit. Manages browser lifecycle and exposes tools.

```python
from definable.browser import BrowserToolkit, BrowserConfig

# Default config
toolkit = BrowserToolkit()

# Custom config
toolkit = BrowserToolkit(config=BrowserConfig(headless=True))

# With action callback
def on_action(event):
    print(f"Browser: {event.action} on {event.selector}")

toolkit = BrowserToolkit(on_action=on_action)
```

**Lifecycle:** Use as async context manager or call `initialize()`/`shutdown()` manually.

```python
# Context manager (recommended)
async with BrowserToolkit() as toolkit:
    agent = Agent(model="openai/gpt-4o", toolkits=[toolkit])
    # ... use agent ...

# Manual lifecycle
toolkit = BrowserToolkit()
await toolkit.initialize()
# ... use ...
await toolkit.shutdown()
```

### Tools Reference

All tools accept either a **ref** (e.g., `"e1"` from `browser_snapshot`) or a **CSS selector** (e.g., `"button.submit"`).

| Category | Tools | Description |
|----------|-------|-------------|
| **Navigation** | `browser_navigate`, `browser_go_back`, `browser_go_forward`, `browser_refresh` | Page navigation |
| **Perception** | `browser_snapshot`, `browser_screenshot` | Read page structure and capture visuals |
| **Page State** | `browser_get_url`, `browser_get_title`, `browser_get_text`, `browser_get_source`, `browser_get_attribute`, `browser_is_visible`, `browser_get_page_info` | Inspect current page |
| **Click** | `browser_click`, `browser_click_if_visible`, `browser_click_by_text` | Click elements |
| **Input** | `browser_type`, `browser_type_slowly`, `browser_press_key`, `browser_press_keys`, `browser_clear_input`, `browser_fill_form`, `browser_set_value`, `browser_set_input_files` | Text entry and form interaction |
| **Selection** | `browser_select_option`, `browser_check`, `browser_uncheck`, `browser_is_checked` | Dropdowns and checkboxes |
| **Scrolling** | `browser_scroll_down`, `browser_scroll_up`, `browser_scroll_to` | Page scrolling |
| **Waiting** | `browser_wait`, `browser_wait_for_element`, `browser_wait_for_text`, `browser_wait_for` | Wait for conditions |
| **Tabs** | `browser_open_tab`, `browser_close_tab`, `browser_get_tabs`, `browser_switch_to_tab` | Tab management |
| **Cookies** | `browser_get_cookies`, `browser_set_cookie`, `browser_clear_cookies` | Cookie management |
| **Storage** | `browser_get_storage`, `browser_set_storage` | localStorage/sessionStorage |
| **Dialogs** | `browser_handle_dialog` | Alert/confirm/prompt handling |
| **DOM** | `browser_highlight`, `browser_remove_elements` | DOM manipulation |
| **JS** | `browser_execute_js` | Execute JavaScript |
| **Diagnostics** | `browser_get_console`, `browser_get_errors`, `browser_get_network` | Debug info |
| **Other** | `browser_hover`, `browser_drag`, `browser_set_geolocation`, `browser_print_to_pdf` | Misc |

### BaseBrowser

Abstract base class for implementing custom browser backends:

```python
from definable.browser import BaseBrowser

class MyBrowser(BaseBrowser):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def snapshot(self, options=None, selector=None, frame_selector=None) -> str: ...
    async def navigate(self, url: str) -> str: ...
    # ... implement all abstract methods ...
```

## Patterns & Recipes

### Agent Workflow Pattern

The recommended pattern for agent browser use:

```
1. browser_snapshot()     ← Understand the page (get element refs)
2. browser_click("e3")   ← Interact using refs
3. browser_snapshot()     ← Re-read after action
4. browser_type("e5", "text")
```

### Removing Overlays

```python
# Agent can dismiss cookie banners before interacting
result = await agent.arun(
    "Go to example.com, remove any cookie banners, then read the main content"
)
# Agent will use: browser_remove_elements(".cookie-notice")
```

### Persistent Login Sessions

```python
config = BrowserConfig(user_data_dir="/tmp/my-chrome-profile")
async with BrowserToolkit(config=config) as toolkit:
    agent = Agent(model="openai/gpt-4o", toolkits=[toolkit])
    # First run: agent logs in and cookies are saved
    # Subsequent runs: already logged in
```

## Gotchas

| Issue | Solution |
|-------|----------|
| `playwright install chromium` not run | Browser launch fails. Run it after `pip install` |
| CDP connection refused | Make sure Chrome is launched with `--remote-debugging-port=9222` |
| Elements not found after navigation | Call `browser_snapshot()` after every page change |
| Timeout errors | Increase `BrowserConfig(timeout=60.0)` |
| Cookie banner blocking interaction | Use `browser_remove_elements(".cookie-banner")` first |

## Related Modules

- **[Agent](../../agent/README.md)** — Browser toolkit plugs into Agent via `toolkits=`
- **[Toolkit](../../toolkit/README.md)** — Base class for all toolkits
- **[Tool](../../tool/README.md)** — Each browser action is a `Function` object
