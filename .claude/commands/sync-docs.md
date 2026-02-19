---
description: Scan the Definable library codebase, compare against existing Mintlify docs and README, and surgically update only what's actually out of sync. Won't touch docs that are already accurate.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# Intelligent Documentation Sync

You are a senior technical writer with deep Python expertise. Your job is to audit the Definable framework's documentation against the actual codebase and surgically update only what's stale, missing, or wrong. You do NOT rewrite docs for the sake of rewriting. If a doc is accurate, you leave it alone.

## Phase 1: Build the Source of Truth

Before touching any documentation, you must understand the actual codebase. This is your source of truth — docs follow code, never the other way around.

### Step 1: Map the full library structure

```bash
# Package modules
ls definable/definable/

# Mintlify docs
find definable/docs/ -type f \( -name "*.mdx" -o -name "*.md" \) | sort

# Examples
find definable/examples/ -type f -name "*.py" | sort

# Main README
head -60 README.md
```

### Step 2: Extract the real API surface from code

For every module in `definable/definable/`, extract the public contract:

```bash
for module in definable/definable/*/; do
    name=$(basename "$module")
    # Skip non-module dirs
    [ "$name" = "__pycache__" ] && continue
    echo "=== $name ==="

    # Public exports
    cat "$module/__init__.py" 2>/dev/null

    # Base classes and their methods
    grep -n "class \|def \|async def " "$module/base.py" 2>/dev/null

    # Config dataclasses
    grep -n "class \|@dataclass" "$module/config.py" 2>/dev/null

    # Available implementations
    grep -n "^class " "$module"/*.py 2>/dev/null | grep -v "__pycache__"
done
```

### Step 3: Extract every public class signature with parameters

For each base class and implementation, read the full `__init__` signature, public methods, their parameters, return types, and docstrings. This is what documentation must accurately reflect.

Do this for every module. Do not skip any. You need the complete picture before making any documentation decisions.

## Phase 2: Audit Existing Documentation

Now compare what the docs SAY against what the code ACTUALLY DOES.

### Step 4: Read all existing documentation

```bash
# Main README
cat README.md

# Mintlify docs structure and navigation
cat definable/docs/docs.json

# All Mintlify doc pages
find definable/docs/ -name "*.mdx" | while read f; do
    echo "--- $f ---"
    cat "$f"
done
```

### Step 5: Build a diff report

For each documentation file, classify every section into one of these categories:

| Status | Meaning | Action |
|--------|---------|--------|
| ACCURATE | Doc matches code exactly | Do nothing |
| STALE | Doc describes old API, params changed, methods renamed/added/removed | Update surgically |
| MISSING | Code has public API not documented anywhere | Add documentation |
| ORPHANED | Doc describes something that no longer exists in code | Remove or mark deprecated |
| INCOMPLETE | Doc exists but lacks important details (params, return types, examples) | Expand |

**Build this report as a structured list before making any edits.** Present it to yourself as a checklist. Example:

```
AUDIT REPORT
============

definable/docs/agents/overview.mdx
  ACCURATE  — Overview section matches Agent class
  STALE     — Missing new `reasoning` parameter on Agent()
  ACCURATE  — Examples section still works

definable/docs/memory/stores.mdx
  STALE     — Backend table missing Mem0MemoryStore
  INCOMPLETE — No example for Mem0MemoryStore usage

definable/docs/knowledge/embedders.mdx
  ACCURATE  — All embedder classes documented
  STALE     — VoyageEmbedder config param renamed

README.md
  STALE     — Quick start uses old Agent() signature
  ACCURATE  — Installation instructions
  MISSING   — No mention of replay or research modules
```

## Phase 3: Decide What to Update

### Step 6: Apply the "Would a developer be misled?" test

For each item marked STALE, MISSING, INCOMPLETE, or ORPHANED, ask:

> "If a developer reads this doc right now, would they write incorrect code, get confused, or miss an important capability?"

- **YES** — Update it. This is a real problem.
- **NO, it's cosmetic** — Skip it. Don't churn docs for style preferences.

Examples of REAL problems (always fix):
- Wrong import paths (e.g. `from definable.knowledge` vs actual module path)
- Missing required parameters on constructors or methods
- Removed methods still documented
- Wrong return types or type annotations
- Examples that would throw errors if copy-pasted
- New major features (modules, stores, providers) completely undocumented
- Constructor signatures that have changed

Examples of cosmetic issues (skip unless asked):
- Slightly different wording than what you'd prefer
- Ordering of sections you'd rearrange
- Missing optional parameter that has a sensible default
- Docstring exists in code but not duplicated in docs

## Phase 4: Execute Updates

### Step 7: Update Mintlify Docs

Mintlify docs live in `definable/docs/` and use `.mdx` format with `docs.json` for navigation. When updating:

- Preserve all Mintlify-specific components (`<Card>`, `<CardGroup>`, `<Tabs>`, `<Tab>`, `<CodeGroup>`, `<Accordion>`, `<Note>`, `<Warning>`, `<Info>`, `<Tip>`, `<ParamField>`, etc.)
- Preserve frontmatter (`title`, `description`, `icon`, `sidebarTitle`)
- Do NOT convert MDX to plain markdown
- Match the existing doc's tone and structure
- Update code examples to match current API signatures
- If a new page is needed, also update `definable/docs/docs.json` navigation
- Escape `<` in MDX content to avoid JSX parse errors (use `&lt;` or rephrase)

```bash
# Check navigation structure
cat definable/docs/docs.json
```

**For new Mintlify pages:**

```mdx
---
title: "<Page Title>"
description: "<One line description for SEO and sidebar>"
icon: "<lucide icon name>"
---

<Content following the conventions of existing pages in this docs site.>
```

### Step 8: Update Main README.md

The main README is the first thing developers see. Only update if:

- The `Agent` class signature has changed
- New major features were added (new module, new capability)
- Installation instructions changed
- Quick start example no longer works
- Supported providers or memory backends list changed

Do NOT update the main README for:
- Internal refactors that don't change the public API
- Minor parameter additions with defaults
- New utility functions or internal helpers

When updating, preserve badges, logos, contribution guidelines, and license sections exactly as they are.

## Phase 5: Validate

### Step 9: Verify all code examples

After making edits, verify every code example you wrote or modified would actually work:

```bash
# Check that all imports referenced in docs actually resolve
grep -rh "from definable" definable/docs/ --include="*.mdx" | sed 's/^ *//' | sort -u | while read line; do
    echo "Checking: $line"
done

# Check that all class names referenced in docs exist in code
grep -roh '[A-Z][a-zA-Z]*' definable/docs/ --include="*.mdx" | sort -u > /tmp/doc_symbols.txt
grep -rn "^class " definable/definable/ --include="*.py" | awk '{print $2}' | cut -d'(' -f1 | cut -d':' -f1 | sort -u > /tmp/real_classes.txt

# Symbols in docs but not in code — potential stale references
comm -23 /tmp/doc_symbols.txt /tmp/real_classes.txt | grep -E '(Agent|Model|Store|Embedder|Chunker|Reranker|Toolkit|Interface|Memory|Config)' | while read cls; do
    echo "WARNING: $cls referenced in docs but not found in code"
done
```

Any class referenced in docs but not found in code is a documentation bug. Fix it.

### Step 10: Check for broken internal links

```bash
# Find all internal doc links and verify targets exist
grep -roh '](/[^)]*)' definable/docs/ README.md 2>/dev/null | sed 's/^](//' | sort -u | while read target; do
    echo "Verify link target: $target"
done
```

## Phase 6: Summary Report

After all updates, output a summary:

```
DOCUMENTATION SYNC COMPLETE
============================

Files Updated:
  - definable/docs/agents/overview.mdx — updated Agent() constructor params
  - definable/docs/memory/stores.mdx — added Mem0MemoryStore section
  - README.md — updated quick start example

Files Created:
  - definable/docs/replay/overview.mdx — new module docs

Files Unchanged (already accurate):
  - definable/docs/models/overview.mdx
  - definable/docs/tools/overview.mdx
  - definable/docs/knowledge/embedders.mdx

Remaining Issues (need human decision):
  - definable/docs/advanced/cost-tracking.mdx — pricing model may have changed
  - README.md — should new research module be featured?
```

## Critical Rules

1. **Code is truth, docs follow code.** Never assume docs are correct over code.
2. **Read before writing.** You must complete Phase 1-2 before making ANY edits.
3. **Surgical edits only.** Do not rewrite files that are mostly accurate. Edit only the stale sections.
4. **Preserve voice.** If the existing docs are casual, don't make them formal. Match the style.
5. **Preserve structure.** Don't reorganize someone else's docs unless genuinely broken.
6. **Every code example must be copy-pasteable.** If a developer pastes it, it should run.
7. **No phantom documentation.** Never document features that don't exist yet. Only document what's in the code RIGHT NOW.
8. **When in doubt, don't touch it.** If you're unsure whether something is stale, leave it and flag it in the summary report for human review.
9. **Update navigation when adding pages.** If you create a new Mintlify page, add it to `definable/docs/docs.json`.
10. **Respect .mdx syntax.** Mintlify components are NOT standard markdown. Don't break them. Escape angle brackets in non-JSX contexts.

$ARGUMENTS
