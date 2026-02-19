---
name: explain-concept
description: Clear explanation of complex topics
version: 1.0.0
tags: [explain, teaching, concepts, learning]
---

## When to Use

Use this skill when explaining technical concepts, algorithms, architectures, or any complex topic to someone who wants to understand it deeply.

## Steps

1. **Assess audience level**: Determine what the reader already knows. Avoid explaining prerequisites they have, and don't skip prerequisites they lack.
2. **Start with the "why"**: Before explaining how something works, explain why it exists — what problem does it solve? What was the alternative before it?
3. **Build incrementally**: Start with the simplest correct mental model, then layer on complexity. Each step should build on the previous one.
4. **Use concrete examples**: For every abstract concept, provide at least one concrete example the reader can trace through mentally.
5. **Highlight trade-offs**: Explain not just what something does, but what it gives up. Every design decision has costs.
6. **Connect to known concepts**: Use analogies to things the reader already understands, but call out where the analogy breaks down.

## Rules

- Never sacrifice correctness for simplicity. If a simplification is misleading, add a caveat.
- Use precise terminology but define it on first use.
- Distinguish between essential complexity (inherent to the problem) and accidental complexity (artifacts of a particular implementation).
- When explaining algorithms or processes, trace through a concrete example step by step.
- Acknowledge common misconceptions and explain why they're wrong.
- If there are competing approaches, briefly mention alternatives and why this one was chosen.

## Output Format

1. **One-liner**: A single sentence explaining what this is and why it matters.
2. **Intuition**: The core idea in simple terms, with an analogy if helpful.
3. **How It Works**: Step-by-step explanation with examples.
4. **Trade-offs**: What this approach gains and gives up.
5. **When to Use / When Not To**: Practical guidance on applicability.
6. **Further Reading**: Pointers to authoritative resources for deeper understanding.
