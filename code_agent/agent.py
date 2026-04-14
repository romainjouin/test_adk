"""
ADK COURSE — "Code Interpreter" Agent
=======================================

This agent writes and executes arbitrary Python code locally via
a custom execute_python tool. It has one single tool that can run
any Python code — making it the ultimate flexible agent.

PEDAGOGICAL CONTRAST:

  Sales Analyst   | "Here are your tools"       (secure, limited)
  NL2SQL          | "Write any SQL query"       (flexible SQL)
  Market Intel    | "Search web + query DB"     (multi-source)
  Code Interpreter| "Write any Python code"     (ultimate flexibility)

Architecture:

  ┌─────────────────────────────────────────────┐
  │  Code Interpreter Agent (Gemini)            │
  │                                             │
  │  tools: [execute_python]                    │
  │  Agent writes code → tool executes it       │
  │  locally → stdout/stderr returned → loop    │
  └──────────────────┬──────────────────────────┘
                     │ subprocess (local Python)
  ┌──────────────────▼──────────────────────────┐
  │  Access to:                                 │
  │  - shop.db (sqlite3 + pandas)               │
  │  - sklearn, matplotlib, numpy               │
  │  - Any installed Python package             │
  │  - Saves charts to static/charts/           │
  └─────────────────────────────────────────────┘
"""

import io
import os
import re
import traceback
from contextlib import redirect_stdout, redirect_stderr

from google.adk.agents import Agent
from google.genai import types as genai_types
from db_context import TODAY

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_BASE_DIR, "shop.db")
_CHARTS_DIR = os.path.join(_BASE_DIR, "static", "charts")
os.makedirs(_CHARTS_DIR, exist_ok=True)

_GLOBALS: dict = {}


def execute_python(code: str) -> dict:
    """Execute Python code locally. Has access to pandas, numpy, sklearn,
    matplotlib, sqlite3 and the shop.db database. State persists between calls.

    Args:
        code: Python code to run. Use print() for output.

    Returns:
        dict with stdout, stderr, status, and chart_urls.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, _GLOBALS)
    except Exception:
        stderr_buf.write(traceback.format_exc())

    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()

    chart_urls = []
    for m in re.finditer(r"CHART_SAVED:(.+?)(?:\n|$)", stdout):
        full_path = m.group(1).strip()
        idx = full_path.find("static/charts/")
        if idx >= 0:
            chart_urls.append("/" + full_path[idx:])

    result = {
        "status": "error" if stderr else "success",
        "stdout": stdout[:5000] if stdout else "",
    }
    if stderr:
        result["stderr"] = stderr[:3000]
    if chart_urls:
        result["chart_urls"] = chart_urls

    return result


_INSTRUCTION = (
    "You are an expert data scientist. You analyze data by writing Python code "
    "and executing it with execute_python. You MUST call execute_python for every "
    "computation — never guess or calculate mentally.\n\n"
    "DATABASE: SQLite at " + _DB_PATH + "\n"
    "Tables: products(id,name,category,price,cost,stock,rating), "
    "customers(id,name,city,signup_date), "
    "orders(id,product_id,quantity,total_price,customer,order_date,status)\n"
    "Statuses: livree=delivered, en cours=pending, annulee=cancelled. "
    "Dates: YYYY-MM-DD HH:MM. Today: " + TODAY + "\n\n"
    "PACKAGES: pandas, numpy, matplotlib, sklearn, sqlite3\n\n"
    "CHARTS: Save to " + _CHARTS_DIR + " using matplotlib. "
    "After plt.savefig, always print CHART_SAVED: followed by the file path.\n\n"
    "RULES:\n"
    "- Always print() results\n"
    "- State persists between execute_python calls\n"
    "- If code fails, read the error, fix it, retry\n"
    "- Show key metrics for ML models\n"
    "- Interpret results in plain English\n"
    "- Suggest follow-up questions\n"
    "- Respond in English"
)


code_agent = Agent(
    name="code_interpreter",
    model="gemini-2.5-flash",
    description=(
        "A code interpreter agent that writes and executes Python code "
        "to analyze data, build ML models, and generate charts."
    ),
    generate_content_config=genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    ),
    instruction=_INSTRUCTION,
    tools=[execute_python],
)

root_agent = code_agent
