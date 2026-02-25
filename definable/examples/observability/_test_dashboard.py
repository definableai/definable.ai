"""Test script: starts the observability dashboard and generates complex events.

Runs diverse agent scenarios to populate all dashboard tabs:
- Multiple tool calls (parallel and sequential)
- Multi-turn conversations with memory
- Structured output
- Many different tools
- Complex multi-step reasoning

Run:
    .venv/bin/python definable/examples/observability/_test_dashboard.py

Then open http://localhost:8000/obs/ in your browser.
"""

import asyncio
import random

from pydantic import BaseModel

from definable.agent import Agent
from definable.agent.runtime.server import AgentServer
from definable.model.openai import OpenAIChat
from definable.tool.decorator import tool


# --- Tools ---


@tool
def get_weather(city: str) -> str:
  """Get the current weather for a city."""
  temps = {
    "Tokyo": "68°F cloudy",
    "London": "55°F rainy",
    "New York": "72°F sunny",
    "Paris": "61°F overcast",
    "Sydney": "78°F clear",
    "Berlin": "50°F foggy",
    "Mumbai": "88°F humid",
  }
  return f"Weather in {city}: {temps.get(city, str(random.randint(50, 90)) + '°F')}"


@tool
def calculate(expression: str) -> str:
  """Evaluate a math expression safely."""
  allowed = set("0123456789+-*/.() ")
  if not all(c in allowed for c in expression):
    return "Error: invalid characters in expression"
  return str(eval(expression))  # noqa: S307


@tool
def lookup_population(country: str) -> str:
  """Look up the population of a country."""
  pops = {
    "US": "331M",
    "UK": "67M",
    "France": "67M",
    "Germany": "83M",
    "Japan": "125M",
    "India": "1.4B",
    "China": "1.4B",
    "Brazil": "214M",
    "Australia": "26M",
    "Canada": "38M",
  }
  return pops.get(country, f"Population data not available for {country}")


@tool
def search_database(query: str) -> str:
  """Search a company database for information."""
  db = {
    "revenue": "Q4 2025 revenue: $12.3M (up 23% YoY). Q3: $10.1M, Q2: $9.5M, Q1: $8.8M.",
    "employees": "Current headcount: 342 across 5 offices (NYC, London, Berlin, Tokyo, Sydney).",
    "products": "3 active products: Platform ($7.2M ARR), API ($3.8M ARR), SDK ($1.3M ARR).",
    "customers": "1,247 active customers. 89% retention rate. Top sectors: fintech, healthcare, SaaS.",
    "churn": "Monthly churn: 1.2%. Top churn reasons: pricing (40%), missing features (35%), support (25%).",
  }
  for key, val in db.items():
    if key in query.lower():
      return val
  return f"No results found for: {query}"


@tool
def translate_text(text: str, target_language: str) -> str:
  """Translate text to target language."""
  translations = {
    "spanish": f"[ES] {text} (traducido al español)",
    "french": f"[FR] {text} (traduit en français)",
    "japanese": f"[JA] {text} (日本語に翻訳)",
    "german": f"[DE] {text} (ins Deutsche übersetzt)",
    "portuguese": f"[PT] {text} (traduzido para português)",
  }
  return translations.get(target_language.lower(), f"[{target_language.upper()}] {text}")


@tool
def generate_report(topic: str, output_format: str = "summary") -> str:
  """Generate an analytical report on a topic."""
  return (
    f"=== Report: {topic} ===\n"
    f"Format: {output_format}\n"
    f"Key findings:\n"
    f"  1. Primary metric increased by 15% quarter-over-quarter\n"
    f"  2. Secondary metric shows seasonal pattern with Q4 peak\n"
    f"  3. Anomaly detected in segment B — needs investigation\n"
    f"  4. Recommendation: increase investment in channel X by 20%\n"
    f"Generated: 2026-02-25T10:00:00Z\n"
    f"Confidence: high"
  )


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
  """Convert between currencies using mock exchange rates."""
  rates = {
    ("USD", "EUR"): 0.92,
    ("USD", "GBP"): 0.79,
    ("USD", "JPY"): 149.5,
    ("EUR", "USD"): 1.09,
    ("GBP", "USD"): 1.27,
    ("JPY", "USD"): 0.0067,
  }
  pair = (from_currency.upper(), to_currency.upper())
  rate = rates.get(pair, 1.0)
  result = amount * rate
  return f"{amount} {from_currency.upper()} = {result:.2f} {to_currency.upper()} (rate: {rate})"


# --- Structured output models ---


class CityAnalysis(BaseModel):
  city: str
  temperature_f: int
  conditions: str
  recommendation: str


class CompanySnapshot(BaseModel):
  revenue: str
  headcount: int
  product_count: int
  customer_count: int
  health_score: float


# --- Agent ---

agent = Agent(
  name="ResearchAssistant",
  model=OpenAIChat(id="gpt-4o-mini"),
  tools=[
    get_weather,
    calculate,
    lookup_population,
    search_database,
    translate_text,
    generate_report,
    convert_currency,
  ],
  instructions=(
    "You are a senior research assistant with access to weather, math, population, "
    "database, translation, reporting, and currency tools. Always use your tools — "
    "never guess or make up data. Be thorough and use multiple tools when needed."
  ),
  memory=True,
  observability=True,
)


async def generate_events():
  """Fire diverse agent runs to populate the dashboard."""
  print("--- Generating agent events ---\n")

  # Run 1: Multi-tool parallel — weather for multiple cities
  r1 = await agent.arun("What's the weather in Tokyo, London, and New York right now?")
  print(f"[1] Weather (3 cities): {(r1.content or '')[:100]}...")

  # Run 2: Math + currency conversion
  r2 = await agent.arun("Calculate 2^16, the square root of 144, and convert $1000 USD to EUR and JPY.")
  print(f"[2] Math + Currency: {(r2.content or '')[:100]}...")

  # Run 3: Multi-turn with memory — population query then follow-up
  r3 = await agent.arun("What's the population of Japan, India, Brazil, and Germany?")
  print(f"[3] Populations: {(r3.content or '')[:100]}...")

  r4 = await agent.arun(
    "Which of those countries I just asked about has the largest population? And what's the total combined population?",
    messages=r3.messages,
  )
  print(f"[4] Follow-up: {(r4.content or '')[:100]}...")

  # Run 4: Database + report (complex multi-step)
  r5 = await agent.arun("Look up our company's revenue, customer data, and churn metrics. Then generate a comprehensive Q4 performance report.")
  print(f"[5] DB + Report: {(r5.content or '')[:100]}...")

  # Run 5: Translation chain (multiple languages)
  r6 = await agent.arun("Translate 'Our quarterly results exceeded all expectations' into Spanish, French, Japanese, and German.")
  print(f"[6] Translations (4 langs): {(r6.content or '')[:100]}...")

  # Run 6: Structured output
  r7 = await agent.arun(
    "Get the weather in Paris and analyze it. Return a structured city analysis.",
    output_schema=CityAnalysis,
  )
  print(f"[7] Structured (CityAnalysis): {(r7.content or '')[:100]}...")

  # Run 7: Complex analysis — multiple DB queries + calculations + report
  r8 = await agent.arun(
    "Search our database for revenue and employee data. Calculate the revenue per "
    "employee. Also convert our total Q4 revenue of $12.3M to EUR and GBP. "
    "Then generate a headcount efficiency report."
  )
  print(f"[8] Complex analysis: {(r8.content or '')[:100]}...")

  # Run 8: Another structured output
  r9 = await agent.arun(
    "Search for our revenue, employee count, products, and customer data. Give me a structured company snapshot with a health score from 0-10.",
    output_schema=CompanySnapshot,
  )
  print(f"[9] Structured (CompanySnapshot): {(r9.content or '')[:100]}...")

  # Run 9: Weather + translate combo
  r10 = await agent.arun("Check the weather in Berlin and Mumbai, then translate the results to Portuguese.")
  print(f"[10] Weather + Translate: {(r10.content or '')[:100]}...")

  # Run 10: Pure calculation chain
  r11 = await agent.arun("Calculate these: (1) 15% of 12300000, (2) 12300000 / 342, (3) 1247 * 0.89, (4) (12300000 - 10100000) / 10100000 * 100")
  print(f"[11] Calc chain: {(r11.content or '')[:100]}...")

  # Run 11: Multi-turn follow-up on the analysis
  r12 = await agent.arun(
    "Based on our earlier conversation about Q4 revenue, generate a report comparing Q4 vs Q3 performance and convert the revenue difference to JPY.",
    messages=r8.messages,
  )
  print(f"[12] Follow-up analysis: {(r12.content or '')[:100]}...")

  print("\n--- Done. 12 runs with diverse tool calls. ---")


async def main():
  import uvicorn

  server = AgentServer(agent, host="0.0.0.0", port=8000, dev=True)
  app = server.create_app()

  print("Dashboard at http://localhost:8000/obs/\n")

  uv_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
  uv_server = uvicorn.Server(uv_config)

  server_task = asyncio.create_task(uv_server.serve())
  await asyncio.sleep(1)

  await generate_events()

  print("\nServer running. Press Ctrl+C to stop.")
  await server_task


if __name__ == "__main__":
  asyncio.run(main())
