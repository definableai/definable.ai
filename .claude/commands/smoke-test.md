---
description: >
  Quick sanity check — imports, install, basic init.
  No full evaluation, no issue filing, no API calls.
  Takes ~30 seconds. Good for "does anything work?"
---

# /smoke-test — Quick Sanity Check

Fast check that the library is functional. No API keys needed. No issues filed.

## Steps

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate 2>/dev/null || true

# 1. Install with all extras
echo "Installing..."
pip install -e ".[readers,serve,jwt,cron,runtime]" 2>&1 | tail -3
if [ $? -ne 0 ]; then
  echo "❌ Install failed"
  exit 1
fi

# 2. Import all modules
echo "Importing modules..."
python -c "
from definable.agent import Agent, AgentConfig, MockModel, create_test_agent
from definable.model import OpenAIChat
from definable.tool.decorator import tool
from definable.skill import Skill, Calculator, DateTime
from definable.knowledge import Document, Knowledge
from definable.vectordb import InMemoryVectorDB
from definable.memory import Memory, InMemoryStore
from definable.agent.guardrail import Guardrails, max_tokens, pii_filter
from definable.agent.events import RunContext, RunStatus
from definable.media import Image, Audio, Video
print('✅ All imports succeeded')
"

# 3. Basic init with MockModel (no API key)
echo "Creating test agent..."
python -c "
from definable.agents import Agent, MockModel, create_test_agent
agent = create_test_agent()
print(f'✅ Agent created: {type(agent).__name__}')
print(f'   Model: {type(agent.model).__name__}')
"

echo "=========================="
echo "  Smoke test complete"
echo "=========================="
```

Report: ✅ All good / ❌ <what failed>

**Does NOT**: Run full tests, file issues, call APIs, or modify any library files.
