# Agent Ideas Research — 2026-02-24

> Stored in memory per CLAUDE.md rules. Cross-reference with competitive-landscape-2026.md.

## Top 10 Agent Categories by Impact (ranked)

1. **Deep Research Agent** — 58% use case demand, native DeepResearch module, score 9/10
2. **AI SDR Agent** — Highest revenue velocity ($500M+ at Cursor), score 7/10
3. **Document Compliance Agent** — Premium legal valuations, strong RAG fit, score 8/10
4. **Code Review Agent** — 63x revenue multiples (dev tools), ClaudeCodeAgent + SubAgents, score 8/10
5. **Customer Support Agent** — #1 deployed use case (26.5%), multi-interface native, score 9/10
6. **Browser Automation Agent** — 78k stars on Browser Use, BrowserToolkit native, score 9/10
7. **Data Analysis Agent** — CFO priority, needs custom tools, score 6/10
8. **Project Management Agent** — 73% SMBs manual, needs external integrations, score 6/10
9. **Email Triage Agent** — Universal pain point, blocked by no email interface, score 5/10
10. **Personal Assistant Agent** — Consumer frontier, MacOS/Desktop native, score 8/10

## Key Market Data Points
- AI agent market: $7.84B (2025) → $52.62B (2030), CAGR 46.3%
- 57.3% of orgs have agents in production (LangChain 2025)
- Top blocker: quality (33%), not cost
- Dev tools: ~63x revenue multiples (highest)
- 90% of AI use cases stuck in pilot mode (McKinsey)

## Critical Library Gaps Identified
1. No email interface (SMTP/IMAP/Gmail)
2. No Slack interface
3. No WhatsApp interface
4. No SQL database connector
5. No calendar integration
6. No charting/visualization
7. No audio transcription
8. No OAuth flow handling
9. No proxy rotation for browser
10. No built-in eval framework

## Innovation Ideas
- SkillMarketplace, AgentCompose (YAML), MemoryGraph, CostGuard, AgentEval

## Claude Code Prompts
- All 10 prompts created in agent-prompts.md (see /home/claude/agent-prompts.md)
- Recommended build order: 1→5→6→10→4→3→2→7→8→9
