---
name: debug-code
description: Systematic debugging methodology
version: 1.0.0
tags: [debug, troubleshoot, errors, bugs]
---

## When to Use

Use this skill when diagnosing bugs, errors, unexpected behavior, or performance issues in code. Applies to runtime errors, logic bugs, test failures, and configuration problems.

## Steps

1. **Reproduce**: Confirm the exact error message, stack trace, or unexpected behavior. Identify the minimal reproduction steps.
2. **Locate**: Trace the error to its origin. Follow the stack trace, identify the failing function, and note the input that triggers the bug.
3. **Hypothesize**: Form 2-3 candidate explanations for the failure. Consider: wrong input, incorrect logic, missing initialization, race condition, dependency issue.
4. **Test hypotheses**: For each hypothesis, identify what evidence would confirm or refute it. Check variable values, execution paths, and preconditions.
5. **Root cause**: Identify the actual root cause (not just the symptom). Ask "why?" until you reach the fundamental issue.
6. **Fix**: Propose a minimal, targeted fix. Explain why it addresses the root cause without introducing side effects.
7. **Verify**: Describe how to verify the fix works and doesn't break other functionality.

## Rules

- Never guess — trace the actual execution path with evidence.
- Distinguish between symptoms and root causes. A crash is a symptom; the unvalidated input that caused it is the root cause.
- Consider recent changes as the most likely source of new bugs.
- Check the simplest explanations first (typos, wrong variable, missing import) before complex ones.
- If the bug is in a dependency or library, say so rather than working around it blindly.

## Output Format

1. **Error**: The exact error or unexpected behavior.
2. **Root Cause**: What is actually going wrong and why.
3. **Fix**: The specific code change needed.
4. **Verification**: How to confirm the fix works.
5. **Prevention**: How to prevent similar bugs in the future (tests, types, assertions).
