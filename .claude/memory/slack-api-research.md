# Slack API Research for SlackInterface

> **Date**: 2026-02-25
> **Purpose**: Production-grade Slack bot/interface for Definable

---

## 1. Slack Bolt for Python

### Framework Overview
- **Package**: `slack-bolt` (PyPI), current version **v1.27.0** (Nov 2025)
- **Python**: 3.7+ required
- **Install**: `pip install slack_bolt`
- **Async dep**: `aiohttp` required for AsyncApp
- **SDK**: `slack-sdk` (underlying Python SDK for Web API, Socket Mode)

### Architecture
- `App` / `AsyncApp` — core application class
- Listener-based: decorators register handlers for events, actions, commands, etc.
- Dependency injection: handler params auto-resolved (ack, say, respond, client, body, event, etc.)
- Middleware chain: global (`app.use()`) and listener-level
- Built-in middleware: RequestVerification, IgnoringSelfEvents, UrlVerification, SslCheck
- Error handler: `@app.error` decorator

### App Constructor Key Parameters
```python
App(
    token="xoxb-...",              # Bot token (single-workspace)
    signing_secret="...",          # HTTP request verification
    # --- OR for multi-workspace ---
    installation_store=store,      # OAuth credential storage
    oauth_settings=settings,       # OAuth flow config
    # --- Behavior ---
    process_before_response=False, # True for FaaS (Lambda, Cloud Functions)
    raise_error_for_unhandled_request=False,
    ignoring_self_events_enabled=True,  # Prevent bot-loop
    # --- Security (all default True) ---
    request_verification_enabled=True,
    token_verification_enabled=True,
    ssl_check_enabled=True,
    url_verification_enabled=True,
)
```

### Listener Types & Decorators
| Decorator | Trigger | Must ack()? |
|-----------|---------|-------------|
| `@app.event("type")` | Events API event | No (auto) |
| `@app.message("pattern")` | Message matching keyword/regex | No (auto) |
| `@app.command("/cmd")` | Slash command | YES (3s) |
| `@app.action("action_id")` | Button click, select, etc. | YES (3s) |
| `@app.shortcut("callback_id")` | Global/message shortcut | YES (3s) |
| `@app.view("callback_id")` | Modal submission/close | YES (3s) |
| `@app.options("action_id")` | External select suggestions | YES |
| `@app.function("func_name")` | Workflow function | YES |

### Listener Arguments (Dependency Injection)
| Arg | Type | Description |
|-----|------|-------------|
| `ack` | callable | Acknowledge receipt (prevents timeout) |
| `say` | callable | Send message to triggering channel |
| `respond` | callable | Ephemeral reply (slash commands, actions) |
| `client` | WebClient | Full Slack API client |
| `body` | dict | Full request payload |
| `event` | dict | Event-specific data |
| `message` | dict | Message object |
| `command` | dict | Slash command data |
| `action` | dict | Interactive action data |
| `view` | dict | Modal view data |
| `payload` | dict | Generic payload |
| `context` | dict | Enriched context |
| `logger` | Logger | App logger |

### AsyncApp Pattern
```python
from slack_bolt.async_app import AsyncApp

app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"],
               signing_secret=os.environ["SLACK_SIGNING_SECRET"])

@app.message("hello")
async def message_hello(message, say):
    await say(f"Hi <@{message['user']}>!")

# Socket Mode (async)
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
await handler.start_async()
```

### Framework Adapters
- **Socket Mode**: `SocketModeHandler` / `AsyncSocketModeHandler`
- **Flask**: `SlackRequestHandler`
- **FastAPI**: `AsyncSlackRequestHandler` (from `slack_bolt.adapter.fastapi.async_handler`)
- **Starlette**, **Sanic**, **Django**, **Tornado**, **Falcon**, etc.

### FastAPI + Socket Mode (Production)
```python
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from contextlib import asynccontextmanager
from fastapi import FastAPI

app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])
handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    await handler.connect_async()
    yield
    await handler.close_async()

api = FastAPI(lifespan=lifespan)
```
**Note**: Socket Mode with FastAPI = single worker only.

---

## 2. Events API

### Outer Envelope Structure
```json
{
  "type": "event_callback",
  "token": "XXYYZZ",
  "team_id": "T123ABC456",
  "api_app_id": "A123ABC456",
  "event": { ... },
  "event_context": "EC123ABC456",
  "event_id": "Ev123ABC456",
  "event_time": 1234567890,
  "authorizations": [{
    "enterprise_id": "E123ABC456",
    "team_id": "T123ABC456",
    "user_id": "U123ABC456",
    "is_bot": false,
    "is_enterprise_install": false
  }],
  "is_ext_shared_channel": false,
  "context_team_id": "T123ABC456"
}
```

### Key Event Types for Chat Interface

#### message (+ subtypes)
- Scopes: `channels:history`, `groups:history`, `im:history`, `mpim:history`
- Channel types: `message.channels`, `message.im`, `message.groups`, `message.mpim`
- Basic payload:
```json
{
  "type": "message",
  "channel": "C123ABC456",
  "user": "U123ABC456",
  "text": "Hello world",
  "ts": "1355517523.000005",
  "channel_type": "channel"  // channel | group | im | mpim
}
```

#### 28 message subtypes
`bot_message`, `me_message`, `message_changed`, `message_deleted`, `message_replied`,
`file_share`, `file_comment`, `file_mention`,
`channel_join`, `channel_leave`, `channel_name`, `channel_topic`, `channel_purpose`,
`channel_archive`, `channel_unarchive`, `channel_convert_to_private`, `channel_convert_to_public`,
`group_join`, `group_leave`, `group_name`, `group_topic`, `group_purpose`,
`group_archive`, `group_unarchive`,
`pinned_item`, `unpinned_item`, `reply_broadcast`, `thread_broadcast`,
`assistant_app_thread`, `document_mention`, `ekm_access_denied`, `reminder_add`,
`channel_posting_permissions`

#### app_mention
- Scope: `app_mentions:read`
- Triggered when bot is @mentioned
```json
{
  "type": "app_mention",
  "user": "U123ABC456",
  "text": "<@U0LAN0Z89> what's up?",
  "ts": "1515449522.000016",
  "channel": "C123ABC456",
  "event_ts": "1515449522000016"
}
```

#### reaction_added / reaction_removed
- Scope: `reactions:read`
```json
{
  "type": "reaction_added",
  "user": "U123ABC456",
  "item": { "type": "message", "channel": "C123ABC456", "ts": "1464196127.000002" },
  "reaction": "thumbsup",
  "item_user": "U222222222",
  "event_ts": "1465244570.336841"
}
```

#### file_shared
- Scope: `files:read`

### Event Delivery
- HTTP POST with `Content-Type: application/json`
- Must respond with HTTP 200 within **3 seconds**
- Retry: 3 attempts with exponential backoff (immediate, 1min, 5min)
- Retry headers: `x-slack-retry-num`, `x-slack-retry-reason`
- Suppress retries: respond with `x-slack-no-retry: 1` header
- Rate: 30,000 events per workspace/app/60min
- **95% failure over 60min = subscriptions temporarily disabled**

---

## 3. Web API Methods

### chat.postMessage
- **Scope**: `chat:write`
- **Rate**: ~1 msg/sec/channel; several hundred/min workspace-wide (Special tier)
- **Key params**:
  | Param | Required | Type | Notes |
  |-------|----------|------|-------|
  | channel | YES | string | Channel ID or user ID |
  | text | Conditional | string | Required if no blocks/attachments; 40k char hard limit |
  | blocks | No | array | Block Kit blocks (max 50 in messages) |
  | attachments | No | array | Legacy; max 100 |
  | thread_ts | No | string | Parent msg ts for threading |
  | reply_broadcast | No | bool | Also post to channel (default: false) |
  | unfurl_links | No | bool | Default: true |
  | unfurl_media | No | bool | Default: true |
  | mrkdwn | No | bool | Default: true |
  | metadata | No | object | {event_type, event_payload} |
  | icon_emoji | No | string | Bot icon override |
  | icon_url | No | string | Bot icon URL override |
  | username | No | string | Bot name override |

- **Response**:
```json
{
  "ok": true,
  "channel": "C123ABC456",
  "ts": "1503435956.000247",
  "message": { "type": "message", "subtype": "bot_message", "text": "...", "ts": "..." }
}
```

### chat.update
- **Scope**: `chat:write`
- **Key constraint**: Only messages posted by authenticated user/bot can be updated
- **Key params**: channel (req), ts (req), text, blocks, attachments, metadata, reply_broadcast
- **Cannot**: update ephemeral messages, replace rich-text blocks with non-rich-text

### files_upload_v2 (Python SDK)
- **Scopes**: `files:write`, `files:read`
- **Deprecation**: `files.upload` deprecated, stopped functioning **March 2025**
- **New 3-step process**:
  1. `files.getUploadURLExternal(filename, length)` -> upload_url + file_id
  2. POST file to upload_url (Content-Type: application/octet-stream)
  3. `files.completeUploadExternal(files=[{id, title}], channel_id)`
- **Python SDK wraps all 3**: `client.files_upload_v2(channel, title, file/content, initial_comment)`
- **File size limit**: 1GB per file (Free plan: 5GB total workspace storage)

```python
# Upload from file path
client.files_upload_v2(
    channel="C123456789",
    title="Report",
    file="./report.pdf",
    initial_comment="Here's the report:",
)

# Upload from content string
client.files_upload_v2(
    channel="C123456789",
    title="Data",
    filename="data.txt",
    content="Hello world",
)
```

### reactions.add
- **Scope**: `reactions:write`
- **Rate**: Tier 3 (50+/min)
- **Params**: channel (req), timestamp (req), name (req, emoji name without colons)

### conversations.history
- **Scope**: `channels:history` / `groups:history` / `im:history` / `mpim:history`
- **Rate**: Tier 3 (50+/min); **Non-Marketplace commercial apps: 1/min, max 15 results** (effective March 2026)
- **Params**: channel (req), cursor, oldest, latest, limit (max 999, default 100), inclusive, include_all_metadata
- **Pagination**: `response_metadata.next_cursor`

### conversations.replies
- **Scope**: same as conversations.history
- **Params**: channel (req), ts (req, parent message ts), cursor, limit (default 15, recommend <=200), oldest, latest
- **Returns**: parent message + all replies, paginated

### conversations.info
- **Scope**: `channels:read` / `groups:read` / `im:read` / `mpim:read`
- **Params**: channel (req), include_locale, include_num_members

### users.info
- **Scope**: `users:read` (+ `users:read.email` for email)
- **Params**: user (req), include_locale
- **Returns**: Full user object (id, name, real_name, profile, tz, is_admin, is_bot, etc.)

---

## 4. Rate Limits

### Tier System (per method, per workspace, per app, per minute)
| Tier | Limit | Examples |
|------|-------|---------|
| Tier 1 | 1+/min | Infrequent access |
| Tier 2 | 20+/min | Most methods |
| Tier 3 | 50+/min | conversations.history, reactions.add |
| Tier 4 | 100+/min | High-volume reads |
| Special | Varies | chat.postMessage (~1/sec/channel) |

### Rate Limit Response
- HTTP 429 Too Many Requests
- `Retry-After` header (seconds to wait)
- Design for 1 req/sec baseline

### Events API Rate
- 30,000 deliveries per workspace/app/60min
- Exceeding triggers `app_rate_limited` event

### Non-Marketplace App Changes (May 2025)
- `conversations.history` and `conversations.replies`: 1 req/min, max 15 results for non-Marketplace commercial apps
- Effective for new apps immediately; existing apps by March 2026

---

## 5. File Object & Media Handling

### File Object Key Fields
```
id, created, name, title, mimetype, filetype, pretty_type, user, size,
mode (hosted|external|snippet|post), is_external, is_public,
url_private, url_private_download, permalink, permalink_public,
thumb_64, thumb_80, thumb_160, thumb_360 (+_w, _h), thumb_480, thumb_720, thumb_960, thumb_1024,
original_w, original_h, image_exif_rotation,
preview, preview_highlight, lines, lines_more, has_rich_preview,
shares, channels, groups, ims, initial_comment,
comments_count, num_stars, is_starred, pinned_to, reactions
```

### Downloading Files
- Use `url_private` or `url_private_download` with Authorization header:
  ```
  Authorization: Bearer xoxb-your-bot-token
  ```
- Bot must be in the channel where file was shared
- `url_private` = view in browser; `url_private_download` = forces download

### File Events
- `file_created`, `file_shared`, `file_unshared`, `file_change`, `file_deleted`, `file_public`

### Image Handling
- Inline display: images uploaded via files API render inline in messages
- External URLs: Use `<url>` in message text (auto-unfurls) or image block
- Thumbnails: auto-generated at multiple sizes (64, 80, 160, 360, 480, 720, 960, 1024)

### Audio/Video
- No native audio message recording in Slack (unlike Telegram)
- Audio/video files can be uploaded as regular files
- Slack Huddles (live audio/video) are separate from messaging API
- Video block type exists in Block Kit for external video embeds

### File Size Limits
- **Per file**: 1 GB
- **Snippets**: 1 MB
- **Free plan total storage**: 5 GB
- **Paid plans**: Varies by plan

---

## 6. Threading Model

### How thread_ts Works
- Every message has a `ts` (timestamp) serving as its unique ID
- To reply in a thread: set `thread_ts` = parent message's `ts`
- **Never use a reply's ts as thread_ts** -- always use the parent
- Parent message gains `reply_count`, `reply_users`, `reply_users_count`, `latest_reply`

### Reply Broadcasting
- `reply_broadcast: true` in `chat.postMessage` makes thread reply visible in channel
- Shows as "Also sent to #channel" indicator
- `thread_broadcast` / `reply_broadcast` message subtype events

### Thread vs Channel Conversations
- Default: replies only visible in thread sidebar
- Broadcast: reply appears in both thread and channel feed
- `conversations.replies(channel, ts)` fetches entire thread
- `conversations.history` returns parent messages (not thread replies)

---

## 7. Rich Messaging (Block Kit)

### Block Types (max 50 per message, 100 per modal/Home tab)
| Block | Purpose |
|-------|---------|
| `section` | Text + optional accessory element |
| `actions` | Interactive elements (buttons, selects, etc.) |
| `context` | Small contextual info (images + text) |
| `divider` | Visual separator |
| `header` | Large bold text |
| `image` | Standalone image |
| `input` | Form input (modals only) |
| `rich_text` | Rich formatted text |
| `file` | File reference |
| `video` | Video embed |
| `table` | Data table |
| `markdown` | Markdown content |

### Section Block
```json
{
  "type": "section",
  "text": { "type": "mrkdwn", "text": "*Bold* and _italic_" },
  "fields": [
    { "type": "mrkdwn", "text": "*Field 1*\nValue" },
    { "type": "mrkdwn", "text": "*Field 2*\nValue" }
  ],
  "accessory": {
    "type": "button",
    "text": { "type": "plain_text", "text": "Click" },
    "action_id": "button_click"
  }
}
```
- text: max 3000 chars
- fields: max 10, each max 2000 chars, rendered in 2 columns
- accessory: one element (button, overflow, datepicker, image, select, etc.)

### Actions Block
```json
{
  "type": "actions",
  "block_id": "actions1",
  "elements": [
    { "type": "button", "text": { "type": "plain_text", "text": "Approve" }, "action_id": "approve", "style": "primary" },
    { "type": "button", "text": { "type": "plain_text", "text": "Reject" }, "action_id": "reject", "style": "danger" }
  ]
}
```
- Max 25 elements per block
- Element types: button, static_select, external_select, users_select, conversations_select, channels_select, multi_select variants, overflow, datepicker, timepicker, radio_buttons, checkboxes

### mrkdwn Formatting
| Syntax | Result |
|--------|--------|
| `*text*` | **Bold** |
| `_text_` | _Italic_ |
| `~text~` | ~~Strikethrough~~ |
| `` `code` `` | Inline code |
| ```` ```code``` ```` | Code block |
| `>quote` | Block quote |
| `<https://url\|label>` | Link with label |
| `<@U12345>` | User mention |
| `<#C12345>` | Channel link |
| `<!here>` | @here |
| `<!channel>` | @channel |
| `<!everyone>` | @everyone |
| `<!subteam^ID>` | User group mention |
| `:emoji_name:` | Emoji |
| `<!date^ts^format\|fallback>` | Date formatting |

### Escaping
- `&` -> `&amp;`
- `<` -> `&lt;`
- `>` -> `&gt;`

---

## 8. Authentication & Permissions

### Token Types
| Token | Prefix | Represents | Usage |
|-------|--------|------------|-------|
| Bot | `xoxb-` | App's bot user | Primary for bots |
| User | `xoxp-` | Workspace member | User-scoped actions |
| App-level | `xapp-` | App across all installs | Socket Mode, org-wide |
| Config | `xoxe-` | Developer config | App manifest APIs |
| Workflow | `xwfp-` | Temp bot subset | Workflow functions (15min TTL) |

### Required Scopes for Full-Featured Bot
```
# Reading
app_mentions:read      # @mention events
channels:history       # Public channel messages
groups:history         # Private channel messages
im:history             # DM messages
mpim:history           # Group DM messages
channels:read          # Channel info
groups:read            # Private channel info
im:read                # DM info
mpim:read              # Group DM info
users:read             # User info
files:read             # File access
reactions:read         # Reaction events

# Writing
chat:write             # Send messages
files:write            # Upload files
reactions:write        # Add reactions

# Optional
commands               # Slash commands
incoming-webhook       # Incoming webhooks
users:read.email       # User email access
```

### OAuth 2.0 Flow
1. App redirects user to `https://slack.com/oauth/v2/authorize?client_id=&scope=&redirect_uri=`
2. User authorizes -> redirected to redirect_uri with `code`
3. App exchanges code for tokens via `oauth.v2.access`
4. Store installation data (team_id, bot_token, user_token, etc.)
5. Scopes are additive across installations (cannot downgrade)

---

## 9. Socket Mode

### How It Works
1. Generate app-level token (`xapp-`) in Basic Information settings
2. Call `apps.connections.open` with app-level token
3. Receive WebSocket URL (`wss://wss.slack.com/link/?ticket=...`)
4. Connect to WebSocket, receive `hello` message
5. All events arrive as WebSocket messages (not HTTP)
6. Acknowledge each event with `envelope_id`

### Benefits
- No public URL needed (works behind firewalls)
- No request signature verification needed (pre-authenticated)
- Bidirectional communication
- Up to 10 concurrent WebSocket connections
- Dynamic URLs (generated at runtime)

### Limitations
- **Cannot list on Slack Marketplace** (public distribution)
- Requires granular permissions (post-Dec 2019 apps)
- Single worker per process (no horizontal scaling of WS connections)

### Envelope Structure
```json
{
  "envelope_id": "unique-id",
  "type": "events_api",
  "payload": { ... event payload ... },
  "accepts_response_payload": true
}
```

### Acknowledgment
```json
{
  "envelope_id": "unique-id",
  "payload": {}
}
```

### Disconnect Reasons
- `link_disabled`: Socket Mode toggled off
- `refresh_requested`: Scheduled connection refresh
- `warning`: 10s notice before disconnect

### Python Implementation
```python
# Sync
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=os.environ["SLACK_BOT_TOKEN"])
handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
handler.start()

# Async
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])
handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
await handler.start_async()
```

### Low-Level SDK Socket Mode
```python
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.web.async_client import AsyncWebClient

client = SocketModeClient(
    app_token=os.environ["SLACK_APP_TOKEN"],
    web_client=AsyncWebClient(token=os.environ["SLACK_BOT_TOKEN"]),
)

async def process(client, req):
    if req.type == "events_api":
        await client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        # process req.payload

client.socket_mode_request_listeners.append(process)
await client.connect()
```

---

## 10. Production Best Practices

### Error Handling
```python
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError

app = AsyncApp(...)

@app.error
async def handle_error(error, body, logger):
    logger.exception(f"Error: {error}")
    logger.info(f"Request body: {body}")

@app.message("hello")
async def greet(message, say, logger):
    try:
        await say(f"Hi <@{message['user']}>!")
    except SlackApiError as e:
        logger.error(f"Slack API error: {e.response['error']}")
```

### Rate Limit Handling
```python
from slack_sdk.errors import SlackApiError
import asyncio

async def send_with_retry(client, channel, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.chat_postMessage(channel=channel, text=text)
        except SlackApiError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 1))
                await asyncio.sleep(retry_after)
            else:
                raise
```

### Message Deduplication
- Use `event_id` from outer envelope (globally unique)
- Cache recent event_ids in Redis/memory with TTL
- Bolt's `IgnoringSelfEvents` middleware prevents bot self-loops

### Idempotency
- Track processed `event_id` + `event_ts` combinations
- Idempotent handlers: check-then-act pattern
- Use `ts` from chat.postMessage response as message reference

### Graceful Shutdown
```python
import signal

async def shutdown(handler):
    await handler.close_async()

# With AsyncSocketModeHandler
loop = asyncio.get_event_loop()
loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(shutdown(handler)))
loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown(handler)))
```

### Retry Logic
- Exponential backoff with jitter
- `slack_sdk` has built-in `RetryHandler`:
  ```python
  from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler
  from slack_sdk.web.async_client import AsyncWebClient

  client = AsyncWebClient(token="xoxb-...")
  client.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=3))
  ```

---

## Design Implications for SlackInterface

### Architecture Decision: Socket Mode vs HTTP
- **Socket Mode recommended for**: Internal tools, development, behind firewalls
- **HTTP recommended for**: Public distribution (Marketplace), multi-process scaling
- **Default**: Socket Mode (simpler, no public URL needed)
- **Option**: Support both via config flag

### Key Design Points
1. **AsyncApp is mandatory** (Definable is all-async)
2. **Event routing**: `@app.message` for DMs, `@app.event("app_mention")` for channel mentions
3. **Threading**: Track `thread_ts` to maintain conversation threads
4. **File handling**: Download via `url_private` + bearer token, upload via `files_upload_v2`
5. **Rate limiting**: Built-in retry handler + per-channel message throttling
6. **Self-event filtering**: Built-in via `ignoring_self_events_enabled`
7. **Multi-workspace**: OAuth flow + InstallationStore for SaaS deployment
8. **Rich responses**: Block Kit for structured output, mrkdwn for text formatting

### Mapping to Definable BaseInterface
| BaseInterface concept | Slack equivalent |
|----------------------|-----------------|
| `start()` | `AsyncSocketModeHandler.start_async()` or mount FastAPI routes |
| `stop()` | `AsyncSocketModeHandler.close_async()` |
| `handle_message` | `@app.message()` / `@app.event("app_mention")` listener |
| `send_response` | `say()` or `client.chat_postMessage()` |
| `user_id` | `event["user"]` |
| `channel_id` | `event["channel"]` |
| `thread_id` | `event["thread_ts"]` or `event["ts"]` |
| `files/media` | `event["files"]` array -> download via url_private |
| `auth` | Bot token scopes + optional AllowlistAuth by user_id |
