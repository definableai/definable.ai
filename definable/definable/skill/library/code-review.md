---
name: code-review
description: Systematic code review with severity-ranked findings
version: 1.0.0
tags: [code, review, quality, bugs]
---

## When to Use

Use this skill when reviewing code changes (diffs, PRs, or complete files) for correctness, maintainability, security, and adherence to best practices.

## Steps

1. **Understand context**: Read the full diff or file. Identify the purpose of the change (feature, bugfix, refactor, etc.).
2. **Check correctness**: Verify logic, edge cases, error handling, and return values. Look for off-by-one errors, null/undefined handling, and race conditions.
3. **Assess security**: Check for injection vulnerabilities (SQL, XSS, command), improper input validation, hardcoded secrets, and insecure defaults.
4. **Evaluate design**: Consider naming clarity, function cohesion, abstraction level, and adherence to existing patterns in the codebase.
5. **Review tests**: Check if changes are covered by tests. Identify missing test cases for edge cases and error paths.
6. **Rank findings**: Assign severity to each finding and present them in priority order.

## Severity Levels

- **Critical**: Bugs that will cause crashes, data loss, or security vulnerabilities in production.
- **Warning**: Issues that may cause problems under certain conditions or violate important conventions.
- **Suggestion**: Improvements to readability, naming, or style that are not strictly required.
- **Nitpick**: Minor style preferences with no functional impact.

## Rules

- Focus on the changed code, not unrelated pre-existing issues (unless they interact with the change).
- Provide concrete fix suggestions, not just problem descriptions.
- Be specific: reference line numbers, function names, and variable names.
- Acknowledge good patterns and well-written code, not just problems.
- If you're uncertain whether something is a bug, say so and explain your reasoning.

## Output Format

For each finding:
```
[SEVERITY] file:line — Short title
Description of the issue and why it matters.
Suggested fix or alternative approach.
```

End with a summary: total findings by severity, overall assessment, and whether the change is ready to merge.
