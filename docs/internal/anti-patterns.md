# Common Anti-Patterns & Fixes

> Load this doc when refactoring or reviewing code quality.

## Anti-Patterns the Agent Must Avoid

### 1. Swallowing Exceptions
```python
# BAD
try:
    result = model.invoke(messages)
except Exception:
    pass

# GOOD
try:
    result = model.invoke(messages)
except ModelProviderError as e:
    log_error(f"Model call failed: {e}")
    raise
```

### 2. Using metadata Instead of meta_data
```python
# BAD — will silently create wrong attribute
Document(content="...", metadata={"source": "x"})

# GOOD
Document(content="...", meta_data={"source": "x"})
```

### 3. Using knowledge=True
```python
# BAD — raises ValueError
Agent(model="gpt-4o", knowledge=True)

# GOOD — must provide vector_db
Agent(model="gpt-4o", knowledge=Knowledge(vector_db=InMemoryVectorDB()))
```

### 4. Deep Inheritance Chains
```python
# BAD
class SpecialAgent(Agent):
    class SpecializedAgent(SpecialAgent):
        ...

# GOOD — use composition
agent = Agent(model=model, middleware=[SpecialBehavior()])
```

### 5. Global Mutable State
```python
# BAD
_global_model = None
def get_model():
    global _global_model
    ...

# GOOD — pass explicitly
def create_agent(model: Model) -> Agent:
    return Agent(model=model)
```

### 6. Mixing Sync and Async in Run Loop
```python
# BAD — sync run() breaks after 2-3 sequential multi-turn calls
for i in range(10):
    result = agent.run(f"turn {i}", messages=result.messages)

# GOOD — use async for multi-turn
async for result in turns:
    result = await agent.arun(prompt, messages=result.messages)
```

## Refactoring Tools
- `jscpd` — code duplication detection
- `ruff check --fix` — auto-fixable lint issues
- `mypy --strict` — type coverage gaps
- Dead code: grep for unused imports, unreachable branches

## When to Refactor
- File > 400 lines → consider splitting
- Function > 50 lines → extract helpers
- Duplicate code across 3+ files → extract shared utility
- Import chain > 3 levels deep → flatten
