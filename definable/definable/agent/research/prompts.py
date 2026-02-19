"""LLM prompt templates for the deep research pipeline."""

DECOMPOSE_PROMPT = """\
You are a research planner. Given a complex question, decompose it into 3-7 \
focused sub-questions that together would fully answer the original question.

Each sub-question should be:
- Specific enough to search for directly
- Non-overlapping with other sub-questions
- Essential to answering the original question

Respond with a JSON array of strings, nothing else.

Example:
Question: "What are the pros and cons of Rust vs Go for building microservices?"
["What are Rust's strengths for microservice development?", \
"What are Go's strengths for microservice development?", \
"How does Rust performance compare to Go in microservice workloads?", \
"What is the developer experience and learning curve for Rust vs Go?", \
"What are the ecosystem and library support differences between Rust and Go for web services?"]

Question: "{query}"
"""

CKU_EXTRACTION_PROMPT = """\
You are an information extraction system. Given a web page and a research question, \
extract all relevant facts as structured data.

Research question: {sub_question}
Original query: {original_query}

Web page content:
---
{page_content}
---

Extract facts that are relevant to the research question. For each fact, provide:
- content: The factual claim (1-2 sentences)
- fact_type: One of "factual", "statistical", "opinion", "definition"
- confidence: 0.0-1.0 how confident you are this is accurate
- entities: List of key entities mentioned
- source_sentence: The original sentence from the page

Also provide:
- relevance_score: 0.0-1.0 how relevant this page is to the question
- page_summary: 1-2 sentence summary of the page
- suggested_followup: A follow-up search query if this page suggests more to explore (or empty string)

Respond with JSON only:
{{
  "facts": [
    {{
      "content": "...",
      "fact_type": "factual",
      "confidence": 0.9,
      "entities": ["entity1", "entity2"],
      "source_sentence": "..."
    }}
  ],
  "relevance_score": 0.8,
  "page_summary": "...",
  "suggested_followup": ""
}}
"""

GAP_ANALYSIS_PROMPT = """\
You are a research coverage analyst. Given a set of research sub-questions and \
the facts collected so far for each, assess whether the research is complete or \
has gaps that need filling.

Original query: {query}

Sub-questions and current coverage:
{coverage_summary}

For each sub-question, assess:
- status: "sufficient" (well-covered), "partial" (some info but gaps), or "missing" (no/very little info)
- confidence: 0.0-1.0 how confident you are in this assessment

For any topic that is "partial" or "missing", suggest 1-2 specific search queries \
to fill the gap.

Respond with JSON only:
{{
  "assessments": [
    {{
      "topic": "sub-question text",
      "status": "sufficient",
      "confidence": 0.9,
      "suggested_queries": []
    }}
  ],
  "new_queries": ["query1", "query2"]
}}
"""

NEEDS_RESEARCH_PROMPT = """\
You are a query classifier. Determine if a user question requires web research \
to answer well, or if it can be answered from general knowledge alone.

Questions that NEED research:
- Current events, recent developments
- Specific statistics, data, or numbers
- Product comparisons with recent info
- Technical documentation for specific versions
- Company or organization details

Questions that DON'T need research:
- General knowledge (math, science basics, definitions)
- Creative tasks (writing, brainstorming)
- Code generation without external context
- Philosophical or opinion questions
- Simple instructions or how-to

Question: "{query}"

Respond with JSON only:
{{"needs_research": true/false, "reason": "brief explanation"}}
"""

SYNTHESIS_PROMPT = """\
You are a research synthesizer. Given organized research facts collected from \
multiple sources, create a concise, well-structured context block that an AI \
assistant can use to answer the original question.

Original query: {query}

Research facts organized by topic:
{organized_facts}

Sources consulted: {source_count}
Total unique facts: {fact_count}

{contradictions_section}

Create a context block in {format} format that:
1. Organizes information by topic/sub-question
2. Highlights key findings and consensus
3. Notes any contradictions or uncertainties
4. Includes source citations (URLs) for key claims
5. Is concise — aim for {max_tokens} tokens or fewer

Return ONLY the formatted context block, ready for injection into a system prompt.
"""
