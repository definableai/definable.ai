# auth

Pluggable HTTP authentication for `agent.serve()` and `AgentServer`.

## Installation

JWTAuth requires an optional dependency:

```bash
pip install 'definable[jwt]'
```

## Quick Start

```python
from definable.agents import Agent
from definable.auth import APIKeyAuth

agent = Agent(model=model, tools=[...])
agent.auth = APIKeyAuth(keys={"sk-my-secret-key"})
agent.serve()
```

## Module Structure

```
auth/
├── __init__.py    # Public API exports (JWTAuth lazy-loaded)
├── base.py        # AuthProvider Protocol, AuthContext dataclass
├── api_key.py     # APIKeyAuth implementation
└── jwt.py         # JWTAuth implementation (requires pyjwt)
```

## API Reference

### AuthProvider (Protocol)

```python
from definable.auth import AuthProvider

class MyAuth:
  def authenticate(self, request) -> Optional[AuthContext]:
    # Return AuthContext on success, None on failure.
    ...
```

Runtime-checkable protocol. The `authenticate` method may be sync or async.

### AuthContext

```python
from definable.auth import AuthContext
```

Dataclass representing an authenticated request:

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | `str` | Authenticated user identifier |
| `metadata` | `Dict[str, Any]` | Additional auth metadata |

### APIKeyAuth

```python
from definable.auth import APIKeyAuth

auth = APIKeyAuth(
  keys={"key-1", "key-2"},  # str or Set[str]
  header="X-API-Key",       # Default header to check
)
```

Checks the configured header first, then falls back to `Authorization` (strips `Bearer ` prefix). Returns `AuthContext` with a hashed `user_id` (SHA256, prefixed `apikey_`).

### JWTAuth

```python
from definable.auth import JWTAuth

auth = JWTAuth(
  secret="my-jwt-secret",
  algorithm="HS256",        # Default
  audience=None,            # Optional audience claim
  issuer=None,              # Optional issuer claim
)
```

Validates JWT from `Authorization: Bearer <token>`. Extracts `user_id` from claims (`sub` > `user_id` > `id`). Returns `AuthContext` with remaining claims as metadata.

## Usage with Agent

```python
from definable.auth import APIKeyAuth, JWTAuth

# API key auth
agent.auth = APIKeyAuth(keys="sk-secret")

# JWT auth
agent.auth = JWTAuth(secret="jwt-secret", audience="my-app")

# Per-trigger auth override
@agent.on(Webhook("/github", auth=APIKeyAuth(keys="gh-secret")))
async def handle_github(event):
  ...

# Disable auth for a specific trigger
@agent.on(Webhook("/public", auth=False))
async def handle_public(event):
  ...
```

## See Also

- `runtime/` — AgentServer that applies auth middleware
- `triggers/` — Per-trigger auth overrides via `Webhook(auth=...)`
