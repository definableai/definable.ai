---
description: "Approve the current plan. Sets plan status to APPROVED so developers can begin."
---

# /approve-plan — Approve the Current Plan

## Execute

1. Read `.agents/handoffs/plan.md` completely
2. Present a concise summary to the user:
   - Goal
   - Number of phases
   - Key design decisions
   - Any open questions for user
3. Ask: "Approve this plan? (The plan is shown above — reply 'yes' to approve, or tell me what to change)"
4. If approved:

```bash
echo "APPROVED" > .agents/queue/plan-status.txt
echo "✅ Plan APPROVED — developers can now run /developer"
```

5. If changes requested: note the changes needed and tell the user to re-run `/planner` with adjustments.
