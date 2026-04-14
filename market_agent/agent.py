"""
ADK COURSE — "Market Intelligence" Agent
=========================================

This agent combines internal sales data with live Google Search
to produce competitive analyses, market comparisons, and trend reports.

PEDAGOGICAL CONTRAST with the other agents:

  Sales Analyst (Toolbox)         |  NL2SQL               |  Market Intelligence (this)
  ────────────────────────────────┼───────────────────────┼─────────────────────────────
  Pre-defined SQL queries         |  Free SQL on shop.db  |  SQL + live web search
  Secure, limited                 |  Flexible, risky      |  Multi-source synthesis
  Internal data only              |  Internal data only   |  Internal + external data

Architecture:

  ┌─────────────────────────────────────────────────────┐
  │  Market Intelligence Agent (Gemini)                 │
  │                                                     │
  │  tools:                                             │
  │   ├─ execute_sql()              read shop.db        │
  │   ├─ google_search_agent        live web search     │
  │   └─ generate_comparison_chart  matplotlib chart    │
  └──────┬──────────────────┬──────────────┬────────────┘
         │                  │              │
  ┌──────▼──────┐  ┌───────▼───────┐  ┌───▼──────────────┐
  │  shop.db    │  │ Google Search │  │ static/charts/   │
  └─────────────┘  └───────────────┘  └──────────────────┘
"""

import json
import os
import sqlite3
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.adk.agents import Agent
from google.adk.tools.google_search_agent_tool import (
    GoogleSearchAgentTool,
    create_google_search_agent,
)
from db_context import PRODUCTS_LIST, STATS_SUMMARY, TODAY

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "shop.db")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# TOOL 1 : Execute a SELECT query on shop.db (read-only)
# ═══════════════════════════════════════════════════════════════════════

def execute_sql(query: str) -> dict:
    """Execute a SQL SELECT query on the e-commerce SQLite database
    and return the results.

    IMPORTANT: use ONLY SELECT queries (read-only).
    Never use INSERT, UPDATE, DELETE, DROP or ALTER.

    Args:
        query: The SQL SELECT query to execute.

    Returns:
        A dict with columns, result rows, and row count.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        results = [dict(row) for row in rows]
        conn.close()
        return {
            "status": "success",
            "sql_executed": query,
            "columns": columns,
            "row_count": len(results),
            "results": results[:50],
        }
    except Exception as e:
        return {
            "status": "error",
            "sql_executed": query,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════
# TOOL 2 : Google Search (via ADK sub-agent wrapper)
# ═══════════════════════════════════════════════════════════════════════

_search_sub_agent = create_google_search_agent(model="gemini-2.5-flash")
google_search_tool = GoogleSearchAgentTool(agent=_search_sub_agent)


# ═══════════════════════════════════════════════════════════════════════
# TOOL 3 : Generate a comparison chart (internal vs. market)
# ═══════════════════════════════════════════════════════════════════════

def generate_comparison_chart(
    title: str,
    labels: str,
    internal_values: str,
    external_values: str,
    value_label: str = "Price (€)",
    chart_type: str = "bar",
) -> dict:
    """Generate a visual comparison chart between internal data and market data.

    Use this after gathering both internal (SQL) and external (web search)
    data to create a side-by-side visual comparison.

    Args:
        title: Chart title (e.g. "Our Prices vs Market Average").
        labels: Comma-separated category labels (e.g. "4K Monitor,Webcam,Headphones").
        internal_values: Comma-separated numeric values for our products
                         (e.g. "499.0,89.99,79.99").
        external_values: Comma-separated numeric values from market data
                         (e.g. "549.0,95.0,85.0").
        value_label: Y-axis label (e.g. "Price (€)", "Market Share (%)", "Rating").
        chart_type: "bar" (default) for grouped bars, "horizontal" for horizontal bars.

    Returns:
        A dict with the chart URL to embed with ![title](chart_url).
    """
    try:
        cats = [l.strip() for l in labels.split(",")]
        internal = [float(v.strip()) for v in internal_values.split(",")]
        external = [float(v.strip()) for v in external_values.split(",")]

        if len(cats) != len(internal) or len(cats) != len(external):
            return {
                "status": "error",
                "error": f"Mismatched lengths: {len(cats)} labels, "
                         f"{len(internal)} internal values, {len(external)} external values.",
            }

        fig, ax = plt.subplots(figsize=(max(8, len(cats) * 1.5), 5))

        if chart_type == "horizontal":
            import numpy as np
            y_pos = np.arange(len(cats))
            h = 0.35
            ax.barh(y_pos - h / 2, internal, h, label="Our Data", color="#4285F4")
            ax.barh(y_pos + h / 2, external, h, label="Market", color="#EA4335")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cats)
            ax.set_xlabel(value_label)
            ax.invert_yaxis()
        else:
            import numpy as np
            x_pos = np.arange(len(cats))
            w = 0.35
            bars1 = ax.bar(x_pos - w / 2, internal, w, label="Our Data", color="#4285F4")
            bars2 = ax.bar(x_pos + w / 2, external, w, label="Market", color="#EA4335")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cats, rotation=30 if len(cats) > 4 else 0, ha="right")
            ax.set_ylabel(value_label)

            for bar in bars1:
                ax.annotate(f"{bar.get_height():.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
            for bar in bars2:
                ax.annotate(f"{bar.get_height():.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.legend()
        ax.grid(axis="y" if chart_type != "horizontal" else "x", alpha=0.3)
        fig.tight_layout()

        filename = f"comparison_{int(time.time())}.png"
        filepath = os.path.join(CHARTS_DIR, filename)
        fig.savefig(filepath, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return {
            "status": "success",
            "chart_url": f"/static/charts/{filename}",
            "title": title,
            "categories": cats,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# THE MARKET INTELLIGENCE AGENT
# ═══════════════════════════════════════════════════════════════════════

market_agent = Agent(
    name="market_intelligence",
    model="gemini-2.5-flash",
    description=(
        "An agent that combines internal e-commerce data with live Google Search "
        "to produce market analyses, competitive benchmarks, and trend reports."
    ),
    instruction=f"""You are a Market Intelligence analyst. You combine internal sales data
with live web research to produce insightful competitive analyses.

## Your unique value
You bridge the gap between internal data and external market context.
A human would need 30+ minutes to gather this manually — you do it in seconds.

## Database schema (SQLite — our internal data)

```sql
CREATE TABLE products (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,     -- Product name
    category    TEXT    NOT NULL,     -- Audio, Périphérique, Vidéo, Accessoire, Écran, Stockage
    price       REAL    NOT NULL,     -- Selling price (incl. tax, in €)
    cost        REAL    NOT NULL,     -- Supplier cost
    stock       INTEGER NOT NULL,     -- Quantity in stock
    rating      REAL    NOT NULL      -- Customer rating (0 to 5)
);

CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    city        TEXT    NOT NULL,
    signup_date TEXT    NOT NULL      -- YYYY-MM-DD
);

CREATE TABLE orders (
    id           INTEGER PRIMARY KEY,
    product_id   INTEGER NOT NULL,
    quantity     INTEGER NOT NULL,
    total_price  REAL    NOT NULL,
    customer     TEXT    NOT NULL,
    order_date   TEXT    NOT NULL,    -- YYYY-MM-DD HH:MM
    status       TEXT    NOT NULL,    -- livrée, en cours, annulée
    FOREIGN KEY (product_id) REFERENCES products(id)
);
```

## Reference data
{STATS_SUMMARY}
- Today's date: {TODAY}
- Margin formula: price - cost
- Margin rate: (price - cost) / price * 100

## Product catalog (EXACT names)
{PRODUCTS_LIST}

## Your 3 tools

### 1. execute_sql(query)
- Queries our internal shop.db database (read-only SELECT)
- Use this FIRST to gather our internal data (prices, sales, margins, trends)
- SQLite date functions: strftime('%Y-%m', order_date), date('{TODAY}', '-12 months')

### 2. google_search_agent (web search)
- Searches the live web via Google Search
- Use this AFTER querying internal data, to get market context
- Good search queries: include product type + "market price 2025-2026" or "market trend" or "benchmark"
- The search returns text with source URLs — always cite them

### 3. generate_comparison_chart(title, labels, internal_values, external_values, value_label, chart_type)
- Creates a side-by-side bar chart comparing our data vs market data
- Use this when you have numeric values from both internal and external sources
- Returns a chart_url — embed it with: ![title](chart_url)

## Your workflow (ALWAYS follow this order)

1. **INTERNAL DATA** — Query shop.db with execute_sql to get our numbers
2. **MARKET RESEARCH** — Search the web with google_search_agent for external context
3. **COMPARE** — If you have comparable numbers, generate a chart with generate_comparison_chart
4. **SYNTHESIZE** — Write a clear analysis that contrasts internal vs external findings

## ABSOLUTE RULE — NO HALLUCINATION

- NEVER invent prices, market shares, or statistics
- Clearly label what comes from "Our data" (SQL results) vs "Market data" (web search)
- If web search doesn't return precise numbers, say so and provide qualitative analysis
- Quote product names EXACTLY as returned by SQL
- Cite web sources with URLs when available

## Response format

Structure your response as:

### Our Data (from database)
[Key metrics from SQL queries]

### Market Context (from web research)
[Findings from Google Search, with source citations]

### Comparison & Analysis
[Your synthesis, chart if applicable]

### Recommendations
[Actionable insights]

## Your rules
- ALWAYS respond in English
- ALWAYS start with execute_sql to get internal data
- ALWAYS search the web for context — that is your differentiator
- When you have comparable numbers, ALWAYS generate a chart
- Embed charts with: ![title](chart_url)
- Cite your web sources
- Suggest a follow-up question at the end
""",
    tools=[execute_sql, google_search_tool, generate_comparison_chart],
)

root_agent = market_agent
