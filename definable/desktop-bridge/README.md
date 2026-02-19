# Definable Desktop Bridge

A Swift companion app that exposes macOS capabilities over a local HTTP API so any Definable agent can control a Mac like a human.

## Requirements

- macOS 13+
- Xcode 15+ or Swift 5.9+
- [Vapor 4](https://vapor.codes/)

## Build & Run

```bash
cd definable/desktop-bridge
swift build -c release
.build/release/DesktopBridge
```

Or run in development mode:

```bash
swift run
```

## First-Run Permissions

The bridge will print its status and warn about missing permissions.
Grant each in **System Settings → Privacy & Security**:

| Permission | Required for |
|-----------|-------------|
| Accessibility | Mouse/keyboard input, AX element inspection |
| Screen Recording | Screenshots, OCR |
| Full Disk Access | Reading files outside your home directory (optional) |

## Security

- Binds to `127.0.0.1:7777` **only** (never 0.0.0.0 — never reachable from the network)
- Bearer token generated on first run and written to `~/.definable/bridge-token` (chmod 600)
- All requests must include `Authorization: Bearer <token>`
- All requests are logged to `~/.definable/bridge.log`

## Token Location

```bash
cat ~/.definable/bridge-token
```

The Python `BridgeClient` reads this file automatically when no token is passed explicitly.

## Endpoints

All endpoints accept `POST` with a JSON body and return JSON.

| Endpoint | Description |
|----------|-------------|
| `POST /health` | Bridge status and permission checks |
| `POST /screen/capture` | Take a screenshot (PNG base64) |
| `POST /screen/ocr` | OCR the screen |
| `POST /screen/find_text` | Find text position on screen |
| `POST /input/click` | Simulate mouse click |
| `POST /input/type` | Type text |
| `POST /input/key` | Press key combination |
| `POST /input/scroll` | Scroll |
| `POST /input/drag` | Click and drag |
| `POST /apps/list` | List running apps |
| `POST /apps/open` | Launch/activate app |
| `POST /apps/quit` | Quit app |
| `POST /apps/activate` | Bring app to foreground |
| `POST /apps/open_url` | Open URL in default browser |
| `POST /windows/list` | List open windows |
| `POST /windows/focus` | Focus a window |
| `POST /windows/resize` | Resize/reposition window |
| `POST /windows/close` | Close a window |
| `POST /ax/get_focused_element` | Get focused UI element |
| `POST /ax/get_ui_tree` | Get app's accessibility tree |
| `POST /ax/find_element` | Find a UI element |
| `POST /ax/perform_action` | Click/press a UI element |
| `POST /ax/set_value` | Set text field value |
| `POST /applescript/run` | Run AppleScript |
| `POST /files/list` | List directory |
| `POST /files/read` | Read file |
| `POST /files/write` | Write file |
| `POST /files/move` | Move/rename file |
| `POST /files/delete` | Delete file (to Trash by default) |
| `POST /files/info` | File metadata |
| `POST /clipboard/get` | Read clipboard |
| `POST /clipboard/set` | Write clipboard |
| `POST /system/info` | System info |
| `POST /system/volume` | Get volume |
| `POST /system/set_volume` | Set volume |
| `POST /system/battery` | Battery status |
| `POST /system/dark_mode` | Get dark mode |
| `POST /system/set_dark_mode` | Set dark mode |
| `POST /system/lock` | Lock screen |
| `POST /notifications/send` | Send macOS notification |

## Using from Python

```python
from definable.agent.interface.desktop import BridgeClient

async with BridgeClient() as client:
    # Token auto-read from ~/.definable/bridge-token
    png = await client.capture_screen()
    await client.click(x=500, y=400)
    text = await client.ocr_screen()
```

Or plug the `MacOS` skill into any agent:

```python
from definable.agent import Agent
from definable.model.openai import OpenAIChat
from definable.skill import MacOS

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    skills=[MacOS()],
    instructions="You can control this Mac.",
)
result = await agent.arun("Open Safari and go to apple.com")
```
