---
description: "Show the current pipeline status — what's done, what's pending, what needs attention."
---

# /pipeline-status — Check Agent Pipeline Status

Read all handoff files and present the current pipeline state.

## Execute

Check each file and report:

```bash
cd "$(git rev-parse --show-toplevel)"
echo "═══════════════════════════════════════"
echo "  AGENT PIPELINE STATUS"
echo "═══════════════════════════════════════"
echo ""

# Task
if [ -f .agents/queue/task.md ]; then
    echo "📋 Task Brief: EXISTS"
    head -5 .agents/queue/task.md
else
    echo "📋 Task Brief: ❌ MISSING — edit .agents/queue/task.md"
fi
echo ""

# Research
if [ -f .agents/handoffs/research.md ]; then
    echo "🔍 Research: ✅ COMPLETE"
else
    echo "🔍 Research: ⏳ PENDING — run /researcher"
fi

# Plan
if [ -f .agents/handoffs/plan.md ]; then
    echo "📐 Plan: ✅ WRITTEN"
    if [ -f .agents/queue/plan-status.txt ]; then
        STATUS=$(cat .agents/queue/plan-status.txt)
        echo "   Plan Status: $STATUS"
    fi
else
    echo "📐 Plan: ⏳ PENDING — run /planner"
fi

# Dev
if [ -f .agents/handoffs/dev-report.md ]; then
    echo "🔨 Development: ✅ COMPLETE"
else
    echo "🔨 Development: ⏳ PENDING — run /developer"
fi

# Test
if [ -f .agents/handoffs/test-report.md ]; then
    echo "🧪 Testing: ✅ COMPLETE"
    # Show verdict
    grep -i "verdict" .agents/handoffs/test-report.md | head -1
else
    echo "🧪 Testing: ⏳ PENDING — run /tester"
fi

# Eval
if [ -f .agents/handoffs/eval-report.md ]; then
    echo "📊 Evaluation: ✅ COMPLETE"
    grep -i "verdict" .agents/handoffs/eval-report.md | head -1
else
    echo "📊 Evaluation: ⏳ PENDING — run /evaluator-agent"
fi

# Docs
if [ -f .agents/handoffs/docs-report.md ]; then
    echo "📝 Documentation: ✅ COMPLETE"
else
    echo "📝 Documentation: ⏳ PENDING — run /docs-agent"
fi

echo ""
echo "═══════════════════════════════════════"
```

After running the bash commands, read any handoff files that exist and give a concise summary of:
1. What's done
2. What needs attention (any bugs, blockers, concerns)
3. What's the next step
