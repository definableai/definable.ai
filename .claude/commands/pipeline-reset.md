---
description: "Reset the agent pipeline. Clears all handoffs and status for a fresh task cycle."
---

# /pipeline-reset — Clear Pipeline for New Task

This clears all agent handoff files and status for a new task cycle.
Previous handoffs are archived to `.agents/archive/`.

## Execute

```bash
cd "$(git rev-parse --show-toplevel)"

# Create archive with timestamp
ARCHIVE_DIR=".agents/archive/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

# Archive current handoffs (if any exist)
if ls .agents/handoffs/*.md 1>/dev/null 2>&1; then
    cp .agents/handoffs/*.md "$ARCHIVE_DIR/"
    echo "📦 Archived handoffs to $ARCHIVE_DIR"
fi

# Archive task brief
if [ -f .agents/queue/task.md ]; then
    cp .agents/queue/task.md "$ARCHIVE_DIR/"
fi

# Clear handoffs
rm -f .agents/handoffs/research.md
rm -f .agents/handoffs/plan.md
rm -f .agents/handoffs/dev-report.md
rm -f .agents/handoffs/test-report.md
rm -f .agents/handoffs/eval-report.md
rm -f .agents/handoffs/docs-report.md

# Clear status
rm -f .agents/queue/plan-status.txt

# Copy fresh task template
cp .agents/queue/task.md.template .agents/queue/task.md

echo ""
echo "✅ Pipeline reset complete"
echo "📝 Edit .agents/queue/task.md with your new task"
echo "🚀 Then launch /researcher in a terminal"
```

Print the final status after running.
