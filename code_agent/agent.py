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
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr

from google.adk.agents import Agent
from db_context import PRODUCTS_LIST, STATS_SUMMARY, TODAY

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_BASE_DIR, "shop.db")
_CHARTS_DIR = os.path.join(_BASE_DIR, "static", "charts")
os.makedirs(_CHARTS_DIR, exist_ok=True)

_GLOBALS: dict = {}


def execute_python(code: str) -> dict:
    """Execute arbitrary Python code and return stdout, stderr, and any chart paths.

    The code runs in the local Python environment with access to all installed
    packages (pandas, numpy, sklearn, matplotlib, sqlite3, etc.) and to the
    e-commerce database at shop.db.

    State is preserved between calls within the same conversation: variables,
    imports, and DataFrames defined in one call are available in the next.

    Args:
        code: The Python code to execute. Must use print() for output.

    Returns:
        A dict with stdout, stderr (if any), and chart_urls (if any charts
        were saved with the CHART_SAVED: marker).
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


code_agent = Agent(
    name="code_interpreter",
    model="gemini-2.5-flash",
    description=(
        "A code interpreter agent that writes and executes Python code "
        "to analyze data, build ML models, and generate charts."
    ),
    instruction=f"""You are an expert Python data scientist. You write and execute
Python code to analyze data, build machine learning models, and create
visualizations.

## How you work
1. Call execute_python with your Python code
2. Read the stdout/stderr output
3. Call execute_python again if needed (state is preserved between calls)
4. If your code fails, read the error, fix it, and try again
5. When done, interpret the results in plain English

## Database access

The e-commerce SQLite database is at:
  DB_PATH = "{_DB_PATH}"

Typical pattern:
  import sqlite3
  import pandas as pd
  conn = sqlite3.connect("{_DB_PATH}")
  df = pd.read_sql("SELECT ...", conn)
  conn.close()
  print(df)

### Schema
  products(id, name, category, price, cost, stock, rating)
  customers(id, name, city, signup_date)
  orders(id, product_id, quantity, total_price, customer, order_date, status)

- Order statuses: livrée (delivered), en cours (pending), annulée (cancelled)
- Date format: YYYY-MM-DD HH:MM
- Today: {TODAY}

### Reference data
{STATS_SUMMARY}

### Product catalog
{PRODUCTS_LIST}

## Available Python packages
- pandas, numpy, matplotlib, sklearn (scikit-learn), sqlite3, json, csv, datetime

## Saving charts

Save matplotlib charts to: {_CHARTS_DIR}

Pattern:
  import matplotlib.pyplot as plt
  import time, os
  fig, ax = plt.subplots(figsize=(10, 6))
  # ... plot ...
  filepath = os.path.join("{_CHARTS_DIR}", "chart_" + str(int(time.time())) + ".png")
  fig.savefig(filepath, dpi=120, bbox_inches="tight")
  plt.close(fig)
  print("CHART_SAVED:" + filepath)

CRITICAL: Always print("CHART_SAVED:" + filepath) after saving a chart.

## Your rules
- ALWAYS respond in English
- ALWAYS use execute_python to run code — never guess results
- ALWAYS print() results (stdout is your only feedback channel)
- ALWAYS use the exact DB_PATH above (absolute path)
- ALWAYS save charts to the CHARTS_DIR path above
- State persists: variables from one execute_python call are available in the next
- When doing ML: explain your approach, show metrics, interpret results
- If code fails: read the error, fix it, retry
- After analysis: interpret results in plain English and suggest follow-up questions
- In your Python code, avoid f-strings — use string concatenation (+) or .format() instead
""",
    tools=[execute_python],
)

root_agent = code_agent
