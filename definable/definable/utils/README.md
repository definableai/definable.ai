# utils

Internal shared utilities — not part of the public API.

## Overview

This module provides common utilities used across the definable package. These are implementation details and may change without notice. Users should prefer the public API from other modules.

## API Reference

Key files:

| File | Purpose |
|------|---------|
| `log.py` | `log_debug`, `log_info`, `log_warning`, `log_error` — structured logging with Rich formatting |
| `supervisor.py` | `supervise_interfaces()` — runs interfaces concurrently with auto-restart and exponential backoff |
| `timer.py` | Timer helpers for performance measurement |
| `http.py` | HTTP request/response utilities |
| `json_schema.py` | JSON Schema generation for tool parameters |
| `serialize.py` | Custom JSON serializer |
| `string.py` | `generate_id()` — deterministic ID generation |
| `media.py` | Media reconstruction helpers (Image, Audio, Video, File from dicts) |
| `functions.py` | Function introspection utilities |
| `tools.py` | Tool-specific helpers |
| `vectordb.py` | `Distance` and `SearchType` enums |
| `reasoning.py` | Reasoning-related utilities |
| `openai.py` | OpenAI-specific helpers |
| `models/` | Model utility sub-package (schema utils, OpenAI response parsing) |

## See Also

- `agents/` — Uses logging, timer, and media utilities
- `interfaces/` — Uses `supervise_interfaces()` for auto-restart
- `tools/` — Uses `json_schema.py` for parameter schema generation
