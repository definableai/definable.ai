---
name: plan-project
description: Project planning with tasks and risk assessment
version: 1.0.0
tags: [planning, project, tasks, management]
---

## When to Use

Use this skill when breaking down a project into tasks, estimating effort, identifying dependencies, assessing risks, or creating an implementation roadmap.

## Steps

1. **Define goals**: Clarify what "done" looks like. Identify must-have vs. nice-to-have outcomes.
2. **Decompose work**: Break the project into concrete, actionable tasks. Each task should be completable independently and verifiable.
3. **Identify dependencies**: Map which tasks block other tasks. Find the critical path (longest chain of dependent tasks).
4. **Assess risks**: For each major component, identify what could go wrong, how likely it is, and what the impact would be.
5. **Prioritize**: Order tasks by dependency constraints, risk reduction, and value delivery. Front-load high-risk items.
6. **Define milestones**: Group tasks into phases with clear deliverables and checkpoints.

## Rules

- Tasks must be specific and verifiable. "Improve performance" is not a task; "Reduce API response time to under 200ms" is.
- Every task should have a clear definition of done.
- Identify assumptions explicitly — plans fail when assumptions are wrong.
- Include buffer for unknowns proportional to the project's novelty (familiar work: 10-20%, novel work: 30-50%).
- Surface dependencies early — blocked tasks are the most common source of delays.
- Separate "what to build" decisions from "how to build" decisions.

## Output Format

1. **Goals**: What the project aims to achieve, with success criteria.
2. **Tasks**: Numbered list with descriptions and definitions of done.
3. **Dependencies**: Which tasks block which (can be a simple list or diagram).
4. **Risks**: Top 3-5 risks with likelihood, impact, and mitigation strategy.
5. **Milestones**: Phased delivery plan with checkpoints.
6. **Assumptions**: Explicit list of things assumed to be true.
