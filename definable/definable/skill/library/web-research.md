---
name: web-research
description: Deep research using web search and source synthesis
version: 1.0.0
tags: [research, web, search, synthesis]
requires_tools: [search_web]
---

## When to Use

Use this skill when you need to research a topic using web sources, verify facts, compare viewpoints, or synthesize information from multiple sources.

## Steps

1. **Clarify the question**: Restate the user's query in specific, searchable terms. Identify what exactly needs to be answered.
2. **Plan search queries**: Generate 2-4 targeted search queries covering different angles of the topic. Prefer specific terms over broad ones.
3. **Search and evaluate**: Execute searches. For each result, assess source credibility (official docs > blogs > forums) and relevance.
4. **Deep dive**: Fetch the most promising URLs to get full context. Look for primary sources, official documentation, or peer-reviewed content.
5. **Cross-reference**: Compare information across sources. Note agreements, contradictions, and gaps.
6. **Synthesize**: Combine findings into a clear, structured answer. Distinguish established facts from opinions or speculation.

## Rules

- Always cite sources with URLs when presenting factual claims.
- Prefer primary sources (official docs, papers, announcements) over secondary summaries.
- Clearly distinguish between well-established facts and uncertain or emerging information.
- If sources conflict, present both viewpoints and explain the disagreement.
- State what you could NOT find or verify, rather than guessing.
- Use at least 2 independent sources before treating a claim as confirmed.

## Output Format

Structure your response as:
1. **Summary**: 2-3 sentence answer to the core question.
2. **Details**: Expanded findings organized by subtopic.
3. **Sources**: Numbered list of URLs used.
4. **Confidence**: Note any areas of uncertainty or gaps in available information.
