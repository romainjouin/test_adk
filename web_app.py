"""
ADK Course — Web Chat Interface
================================

Three agents available via tab selector:
  - Sales Analyst         : pre-defined SQL tools via MCP Toolbox (secure)
  - NL2SQL                : generates SQL on the fly (flexible)
  - Market Intelligence   : internal data + live Google Search (wow factor)

Usage:
    python web_app.py
    → open http://localhost:8080
"""

import json
import uuid
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from sales_agent.agent import root_agent as sales_agent
from nl2sql_agent.agent import nl2sql_agent
from market_agent.agent import market_agent
from code_agent.agent import code_agent

load_dotenv("sales_agent/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("adk-demo")

session_service = InMemorySessionService()
runners: dict[str, Runner] = {}

# ─── Conversation memory ────────────────────────────────────────────
# Per-session list of {"q": "user question", "a": "short answer summary"}.
# Injected as context prefix into every subsequent user message so that
# the agent (and any sub-agent it delegates to) has explicit context.
_conv_memory: dict[str, list[dict]] = {}  # key = f"{agent_key}:{session_id}"

MAX_MEMORY_TURNS = 10  # keep the last N exchanges to avoid prompt bloat


def _build_context_prefix(mem_key: str) -> str:
    """Build a markdown context block from previous exchanges."""
    history = _conv_memory.get(mem_key, [])
    if not history:
        return ""
    lines = ["[Conversation context — previous exchanges in this session]"]
    for i, turn in enumerate(history[-MAX_MEMORY_TURNS:], 1):
        lines.append(f"{i}. Q: {turn['q']}")
        lines.append(f"   A: {turn['a']}")
    lines.append("[End of context — current question follows]\n")
    return "\n".join(lines)


def _record_turn(mem_key: str, question: str, answer: str):
    """Store a summarised Q/A turn in memory."""
    summary = answer[:300].replace("\n", " ").strip()
    if mem_key not in _conv_memory:
        _conv_memory[mem_key] = []
    _conv_memory[mem_key].append({"q": question, "a": summary})
    if len(_conv_memory[mem_key]) > MAX_MEMORY_TURNS:
        _conv_memory[mem_key] = _conv_memory[mem_key][-MAX_MEMORY_TURNS:]

AGENTS = {
    "sales": {
        "agent": sales_agent,
        "app_name": "sales_web",
        "label": "Sales Analyst",
        "icon": "📊",
        "subtitle": "Pre-defined SQL tools via MCP Toolbox",
        "welcome": "Hello! I'm your e-commerce analyst. I use **pre-defined tools** to query the database.\n\nAsk me about products, sales, margins, or stock levels!",
        "suggestions": [
            "What is our best-seller?",
            "Analyze product margins",
            "Any products low on stock?",
            "Show the last 5 orders",
            "Which products are in the Audio category?",
        ],
    },
    "nl2sql": {
        "agent": nl2sql_agent,
        "app_name": "nl2sql_web",
        "label": "NL2SQL",
        "icon": "🧠",
        "subtitle": "Generates SQL on the fly from natural language",
        "welcome": "Hello! I'm the **NL2SQL** agent. Ask me any question about the database, I'll convert it to SQL and execute it.\n\nI'll show you the generated SQL query before the results!",
        "suggestions": [
            "Which customer spent the most?",
            "Revenue by category",
            "Create a monthly_kpi table with revenue and order count per month",
            "Order cancellation rate",
            "List existing feature tables",
        ],
    },
    "market": {
        "agent": market_agent,
        "app_name": "market_web",
        "label": "Market Intel",
        "icon": "🌍",
        "subtitle": "Internal data + live Google Search",
        "welcome": "Hello! I'm the **Market Intelligence** agent. I combine your internal sales data with **live web search** to produce competitive analyses and market reports.\n\nAsk me to compare your products to the market, spot trends, or benchmark your prices!",
        "suggestions": [
            "Are our screen prices competitive compared to the market?",
            "What are the current trends in the webcam market?",
            "Benchmark our top 5 products against competitors",
            "Which product category should we invest in next?",
            "How do our margins compare to industry averages for tech accessories?",
        ],
    },
    "code": {
        "agent": code_agent,
        "app_name": "code_web",
        "label": "Code Interpreter",
        "icon": "💻",
        "subtitle": "Writes & runs Python code autonomously",
        "welcome": "Hello! I'm the **Code Interpreter** agent. I have no pre-built tools — I write and execute **Python code** on the fly.\n\nI can access your database with pandas, train ML models with scikit-learn, and generate charts with matplotlib.\n\nAsk me anything — I'll write the code to answer it!",
        "suggestions": [
            "Segment customers by purchasing behavior using KMeans",
            "Plot monthly revenue over the last 2 years",
            "Which factors most influence order cancellation? Train a classifier",
            "Create an RFM analysis of our customers",
            "Show a heatmap of sales by category and month",
        ],
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for key, cfg in AGENTS.items():
        runners[key] = Runner(
            agent=cfg["agent"],
            app_name=cfg["app_name"],
            session_service=session_service,
        )
    yield


app = FastAPI(lifespan=lifespan)

import os
_static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(os.path.join(_static_dir, "charts"), exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    agent: str = "sales"


TOOL_LABELS = {
    # Market Intelligence tools
    "google_search_agent": "🌐 Searching the web",
    "generate_comparison_chart": "📊 Generating comparison chart",
    # Code Interpreter
    "execute_python": "💻 Running Python code",
    # Toolbox tools
    "search-products-by-category": "🔎 Search by category",
    "search-products-by-name": "🔎 Search by name",
    "get-top-products": "⭐ Top products",
    "get-sales-by-product": "📊 Sales by product",
    "get-recent-orders": "🕐 Recent orders",
    "get-low-stock-products": "⚠️ Low stock",
    "get-margin-analysis": "💰 Margin analysis",
    # Agent delegation
    "nl2sql_analyst": "🧠 Delegating to SQL expert",
    # NL2SQL
    "execute_sql": "🗄️ Executing SQL",
    # Feature engineering
    "create_feature_table": "🏗️ Creating feature table",
    "add_column": "➕ Adding column",
    "insert_query_results": "📥 Inserting results",
    "list_feature_tables": "📋 Listing feature tables",
    "query_feature_table": "🔍 Querying feature table",
    "export_table_csv": "📤 CSV export",
}


def _enrich_tool_call(evt_data: dict, agent_key: str):
    """Add human-readable info to tool_call SSE events."""
    name = evt_data["name"]
    args = evt_data["args"]
    label = TOOL_LABELS.get(name)

    if label:
        evt_data["rich_label"] = label

    if name == "execute_sql":
        evt_data["sql_query"] = args.get("query")
    elif name == "nl2sql_analyst":
        evt_data["rich_detail"] = args.get("request", "")[:300]
    elif name == "execute_python":
        code_text = args.get("code", "")
        evt_data["code"] = code_text
    elif name == "google_search_agent":
        evt_data["rich_detail"] = args.get("request", "")[:300]
    elif name == "generate_comparison_chart":
        evt_data["rich_detail"] = f"{args.get('title', '?')}  •  {args.get('labels', '')[:100]}"
    elif name == "create_feature_table":
        evt_data["rich_detail"] = f"Table : {args.get('table_name', '?')}  •  Colonnes : {args.get('columns', '?')}"
    elif name == "add_column":
        evt_data["rich_detail"] = f"Table : {args.get('table_name', '?')}  •  +{args.get('column_name', '?')} ({args.get('column_type', 'REAL')})"
    elif name == "insert_query_results":
        evt_data["rich_detail"] = f"→ {args.get('table_name', '?')}"
        evt_data["sql_query"] = args.get("source_query")
    elif name == "query_feature_table":
        tbl = args.get("table_name", "?")
        q = args.get("query", "")
        evt_data["rich_detail"] = f"Table: {tbl}" + (f"  •  {q[:200]}" if q else " (all rows)")
        if q:
            evt_data["sql_query"] = q
    elif name == "export_table_csv":
        evt_data["rich_detail"] = f"Table : {args.get('table_name', '?')}"
    elif name.startswith("search-products"):
        kw = args.get("keyword") or args.get("category") or ""
        evt_data["rich_detail"] = f'"{kw}"'
    elif name == "get-recent-orders":
        evt_data["rich_detail"] = f"last {args.get('limit', '?')}"
    elif name == "get-low-stock-products":
        evt_data["rich_detail"] = f"threshold ≤ {args.get('threshold', '?')}"
    elif name == "get-top-products":
        evt_data["rich_detail"] = f"top {args.get('limit', '?')}"


def _enrich_tool_result(evt_data: dict, resp_data):
    """Add human-readable info to tool_result SSE events."""
    name = evt_data["name"]
    label = TOOL_LABELS.get(name, name)

    if not isinstance(resp_data, dict):
        text = str(resp_data)
        preview = text[:500]
        evt_data["rich_summary"] = f"✅ {label}"
        evt_data["rich_detail"] = preview
        return

    status = resp_data.get("status", "")
    if status == "error" and name != "execute_python":
        evt_data["rich_summary"] = f"❌ Error"
        evt_data["rich_detail"] = resp_data.get("error", "?")[:300]
        return

    if name == "execute_python":
        stdout = resp_data.get("stdout", "")
        stderr = resp_data.get("stderr", "")
        charts = resp_data.get("chart_urls", [])
        if stderr:
            evt_data["rich_summary"] = "❌ Code execution error"
            evt_data["code_stderr"] = stderr[:2000]
        else:
            evt_data["rich_summary"] = "✅ Code executed"
        if stdout:
            evt_data["code_stdout"] = stdout[:3000]
        if charts:
            evt_data["chart_urls"] = charts

    elif name == "google_search_agent":
        result_text = str(resp_data) if not isinstance(resp_data, dict) else resp_data.get("result", str(resp_data))
        evt_data["rich_summary"] = f"✅ Web search complete"
        preview = str(result_text)[:400]
        evt_data["rich_detail"] = preview

    elif name == "generate_comparison_chart":
        evt_data["rich_summary"] = f"✅ Chart generated"
        evt_data["chart_url"] = resp_data.get("chart_url", "")
        cats = resp_data.get("categories", [])
        if cats:
            evt_data["rich_detail"] = f"Comparing: {', '.join(cats)}"

    elif name == "execute_sql":
        row_count = resp_data.get("row_count", 0)
        cols = resp_data.get("columns", [])
        evt_data["rich_summary"] = f"✅ {row_count} rows  •  {', '.join(cols[:5])}"
        results = resp_data.get("results", [])
        if results:
            preview_rows = results[:3]
            evt_data["rich_detail"] = json.dumps(preview_rows, ensure_ascii=False)[:400]

    elif name == "nl2sql_analyst":
        result_text = ""
        if isinstance(resp_data, dict) and "result" in resp_data:
            result_text = str(resp_data["result"])
        elif isinstance(resp_data, str):
            result_text = resp_data
        else:
            result_text = json.dumps(resp_data, ensure_ascii=False)
        evt_data["rich_summary"] = f"✅ SQL expert responded"
        evt_data["rich_detail"] = result_text[:500]

    elif name == "create_feature_table":
        cols = resp_data.get("columns", [])
        evt_data["rich_summary"] = f"✅ Table '{resp_data.get('table_name', '?')}' created"
        evt_data["rich_detail"] = ", ".join(cols)

    elif name == "add_column":
        evt_data["rich_summary"] = f"✅ Column added: {resp_data.get('added_column', '?')}"
        cols = resp_data.get("columns", [])
        evt_data["rich_detail"] = ", ".join(cols)

    elif name == "insert_query_results":
        inserted = resp_data.get("rows_inserted", 0)
        total = resp_data.get("total_rows", "?")
        evt_data["rich_summary"] = f"✅ {inserted} rows inserted ({total} total)"
        preview = resp_data.get("preview", [])
        if preview:
            evt_data["rich_detail"] = json.dumps(preview[:3], ensure_ascii=False)[:400]

    elif name == "list_feature_tables":
        tables = resp_data.get("tables", [])
        evt_data["rich_summary"] = f"✅ {len(tables)} feature table(s)"
        if tables:
            parts = [f"{t['table_name']} ({t.get('row_count', '?')} rows)" for t in tables]
            evt_data["rich_detail"] = "  •  ".join(parts)

    elif name == "query_feature_table":
        row_count = resp_data.get("row_count", 0)
        tbl = resp_data.get("table_name", "?")
        truncated = resp_data.get("truncated", False)
        evt_data["rich_summary"] = f"✅ {row_count} row(s) from '{tbl}'" + (" (truncated)" if truncated else "")
        results = resp_data.get("results", [])
        if results:
            evt_data["rich_detail"] = json.dumps(results[:3], ensure_ascii=False)[:400]

    elif name == "export_table_csv":
        url = resp_data.get("download_url", "")
        row_count = resp_data.get("row_count", 0)
        evt_data["rich_summary"] = f"✅ Export: {row_count} rows"
        if url:
            evt_data["download_url"] = url
            evt_data["download_name"] = resp_data.get("table_name", "features") + ".csv"
        evt_data["rich_detail"] = resp_data.get("message", "")

    elif name.startswith(("search-", "get-")):
        if isinstance(resp_data, dict) and "result" in resp_data:
            raw = resp_data["result"]
            if isinstance(raw, list):
                evt_data["rich_summary"] = f"✅ {len(raw)} result(s)"
            elif raw is None:
                evt_data["rich_summary"] = "⚠️ No results"
            else:
                evt_data["rich_summary"] = f"✅ Result received"
        else:
            evt_data["rich_summary"] = f"✅ Result received"

    else:
        preview = json.dumps(resp_data, ensure_ascii=False)[:500]
        evt_data["rich_summary"] = f"✅ {label}"
        evt_data["rich_detail"] = preview


def _log_tool_call(name: str, args: dict, agent_label: str):
    icon = TOOL_LABELS.get(name, f"🔧 {name}").split(" ")[0]
    label = TOOL_LABELS.get(name, name)
    if name == "execute_sql":
        log.info("  %s [%s] SQL → %s", icon, agent_label, args.get("query", "")[:250])
    elif name == "nl2sql_analyst":
        log.info("  %s [%s] → \"%s\"", icon, agent_label, args.get("request", "")[:200])
    elif name == "execute_python":
        code = args.get("code", "")
        lines = code.count("\n") + 1
        log.info("  %s [%s] Python code (%d lines):\n%s", icon, agent_label, lines, code[:500])
    elif name == "google_search_agent":
        log.info("  %s [%s] web search → \"%s\"", icon, agent_label, args.get("request", "")[:200])
    elif name == "generate_comparison_chart":
        log.info("  %s [%s] chart → %s  labels=%s", icon, agent_label, args.get("title", ""), args.get("labels", "")[:100])
    elif name == "create_feature_table":
        log.info("  %s [%s] table=%s  cols=%s", icon, agent_label, args.get("table_name"), args.get("columns", "")[:150])
    elif name == "add_column":
        log.info("  %s [%s] table=%s  +%s (%s)", icon, agent_label, args.get("table_name"), args.get("column_name"), args.get("column_type", "REAL"))
    elif name == "insert_query_results":
        log.info("  %s [%s] → %s  SQL=%s", icon, agent_label, args.get("table_name"), args.get("source_query", "")[:200])
    elif name == "list_feature_tables":
        log.info("  %s [%s] Listing feature tables", icon, agent_label)
    elif name == "query_feature_table":
        log.info("  %s [%s] table=%s  SQL=%s", icon, agent_label, args.get("table_name"), args.get("query", "(all)")[:200])
    elif name == "export_table_csv":
        log.info("  %s [%s] Export table=%s", icon, agent_label, args.get("table_name"))
    elif name.startswith("search-products"):
        kw = args.get("keyword") or args.get("category") or ""
        log.info("  %s [%s] → \"%s\"", icon, agent_label, kw)
    elif name.startswith(("get-", "search-")):
        log.info("  %s [%s] %s → %s", icon, agent_label, label, json.dumps(args, ensure_ascii=False)[:100])
    else:
        log.info("  🔧 [%s] %s → %s", agent_label, name, json.dumps(args, ensure_ascii=False)[:200])


def _log_tool_result(name: str, resp_data):
    icon = TOOL_LABELS.get(name, f"📦 {name}").split(" ")[0]

    if not isinstance(resp_data, dict):
        text = str(resp_data)[:200]
        if name == "nl2sql_analyst":
            log.info("  📦 %s response (%d chars)", name, len(str(resp_data)))
        else:
            log.info("  📦 %s → %s", name, text)
        return

    status = resp_data.get("status", "ok")
    if status == "error":
        log.warning("  ❌ %s → ERREUR : %s", name, resp_data.get("error", "?")[:200])
        return

    if name == "execute_python":
        stdout = resp_data.get("stdout", "")[:200]
        stderr = resp_data.get("stderr", "")
        charts = resp_data.get("chart_urls", [])
        if stderr:
            log.warning("  ❌ execute_python → ERROR: %s", stderr[:200])
        else:
            log.info("  📦 execute_python → %s%s", stdout, f"  charts={charts}" if charts else "")
    elif name == "google_search_agent":
        result_text = str(resp_data)[:200] if not isinstance(resp_data, dict) else str(resp_data.get("result", ""))[:200]
        log.info("  📦 %s → %s", name, result_text)
    elif name == "generate_comparison_chart":
        log.info("  📦 %s → %s", name, resp_data.get("chart_url", "?"))
    elif name == "execute_sql":
        log.info("  📦 %s → %d lignes, colonnes: %s",
                 name, resp_data.get("row_count", 0), resp_data.get("columns", []))
    elif name == "nl2sql_analyst":
        result = resp_data.get("result", str(resp_data))
        log.info("  📦 %s → response (%d chars)", name, len(str(result)))
    elif name == "create_feature_table":
        log.info("  📦 %s → table '%s' created, columns: %s", name, resp_data.get("table_name"), resp_data.get("columns"))
    elif name == "add_column":
        log.info("  📦 %s → column added: %s", name, resp_data.get("added_column"))
    elif name == "insert_query_results":
        log.info("  📦 %s → %d rows inserted (%s total)", name, resp_data.get("rows_inserted", 0), resp_data.get("total_rows", "?"))
    elif name == "list_feature_tables":
        tables = resp_data.get("tables", [])
        names = [t.get("table_name", "?") for t in tables]
        log.info("  📦 %s → %d table(s): %s", name, len(tables), ", ".join(names))
    elif name == "query_feature_table":
        log.info("  📦 %s → %d row(s) from '%s'", name, resp_data.get("row_count", 0), resp_data.get("table_name", "?"))
    elif name == "export_table_csv":
        log.info("  📦 %s → %s (%d rows)", name, resp_data.get("download_url", "?"), resp_data.get("row_count", 0))
    elif name.startswith(("search-", "get-")):
        raw = resp_data.get("result")
        if isinstance(raw, list):
            log.info("  📦 %s → %d result(s)", name, len(raw))
        elif raw is None:
            log.info("  📦 %s → no results", name)
        else:
            log.info("  📦 %s → result received", name)
    else:
        log.info("  📦 %s → %s", name, json.dumps(resp_data, ensure_ascii=False)[:200])


@app.post("/api/chat")
async def chat(req: ChatRequest):
    agent_key = req.agent if req.agent in runners else "sales"
    runner = runners[agent_key]
    app_name = AGENTS[agent_key]["app_name"]
    agent_label = AGENTS[agent_key]["label"]

    session_id = req.session_id or str(uuid.uuid4())

    existing = await session_service.get_session(
        app_name=app_name, user_id="web_user", session_id=session_id
    )
    if not existing:
        await session_service.create_session(
            app_name=app_name, user_id="web_user", session_id=session_id
        )

    log.info("━" * 60)
    log.info("💬 [%s] Question : %s", agent_label, req.message)

    # ── Inject conversation context into the message ──
    mem_key = f"{agent_key}:{session_id}"
    context_prefix = _build_context_prefix(mem_key)
    if context_prefix:
        enriched_text = context_prefix + req.message
        n_turns = len(_conv_memory.get(mem_key, []))
        log.info("🧠 [%s] Context injected (%d previous exchanges)", agent_label, n_turns)
    else:
        enriched_text = req.message

    message = types.Content(
        role="user",
        parts=[types.Part(text=enriched_text)],
    )

    user_question = req.message  # keep original for memory
    _context_turns = len(_conv_memory.get(mem_key, []))

    async def event_stream():
        import time
        t0 = time.time()
        final_answer = ""

        if _context_turns > 0:
            evt = {"type": "context_injected", "turns": _context_turns}
            yield f"data: {json.dumps(evt)}\n\n"

        async for event in runner.run_async(
            user_id="web_user", session_id=session_id, new_message=message
        ):
            function_calls = event.get_function_calls()
            if function_calls:
                for fc in function_calls:
                    args = dict(fc.args)
                    _log_tool_call(fc.name, args, agent_label)

                    sse_args = {k: v for k, v in args.items() if k != "data_json"}
                    evt_data = {"type": "tool_call", "name": fc.name, "args": sse_args}
                    _enrich_tool_call(evt_data, agent_key)
                    yield f"data: {json.dumps(evt_data, ensure_ascii=False)}\n\n"

            function_responses = event.get_function_responses()
            if function_responses:
                for fr in function_responses:
                    resp_data = fr.response if hasattr(fr, "response") else str(fr)
                    evt_data = {"type": "tool_result", "name": fr.name}
                    _enrich_tool_result(evt_data, resp_data)

                    _log_tool_result(fr.name, resp_data)
                    yield f"data: {json.dumps(evt_data, ensure_ascii=False)}\n\n"

            if event.is_final_response():
                if event.content and event.content.parts:
                    text = "".join(p.text for p in event.content.parts if hasattr(p, "text") and p.text)
                    if not text:
                        continue
                    final_answer = text
                    elapsed = time.time() - t0
                    preview = text[:150].replace("\n", " ")
                    log.info("✅ [%s] Response (%0.1fs): %s…", agent_label, elapsed, preview)
                    yield f"data: {json.dumps({'type': 'answer', 'text': text, 'session_id': session_id}, ensure_ascii=False)}\n\n"

        # Record this exchange in conversation memory
        if final_answer:
            _record_turn(mem_key, user_question, final_answer)
            log.info("🧠 Memory updated (%d exchanges)", len(_conv_memory.get(mem_key, [])))

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/agents")
async def list_agents():
    return {k: {kk: vv for kk, vv in v.items() if kk != "agent"} for k, v in AGENTS.items()}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ADK Demo — AI Agents</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242736;
    --border: #2e3245;
    --text: #e4e6f0;
    --text-muted: #8b8fa3;
    --accent: #6c63ff;
    --accent-glow: rgba(108, 99, 255, 0.15);
    --tool-bg: #1c2333;
    --tool-border: #2a3a5c;
    --success: #34d399;
    --sql-accent: #f59e0b;
    --sql-glow: rgba(245, 158, 11, 0.12);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  header {
    padding: 12px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--surface);
  }

  .agent-switcher {
    display: flex;
    gap: 6px;
    background: var(--bg);
    padding: 4px;
    border-radius: 12px;
  }

  .agent-tab {
    padding: 8px 16px;
    border: none;
    border-radius: 10px;
    background: transparent;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: inherit;
  }
  .agent-tab:hover { color: var(--text); }
  .agent-tab.active {
    background: var(--surface2);
    color: var(--text);
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
  }
  .agent-tab.active[data-agent="sales"] { border-bottom: 2px solid var(--accent); }
  .agent-tab.active[data-agent="nl2sql"] { border-bottom: 2px solid var(--sql-accent); }
  .agent-tab.active[data-agent="market"] { border-bottom: 2px solid var(--success); }
  .agent-tab.active[data-agent="code"] { border-bottom: 2px solid #a78bfa; }

  .header-info {
    margin-left: auto;
    text-align: right;
  }
  .header-info .label { font-size: 13px; font-weight: 600; }
  .header-info .sub { font-size: 11px; color: var(--text-muted); }

  #chat-container {
    flex: 1;
    position: relative;
    overflow: hidden;
  }
  .chat-panel {
    position: absolute;
    inset: 0;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    visibility: hidden;
    pointer-events: none;
  }
  .chat-panel.active-panel {
    visibility: visible;
    pointer-events: auto;
  }

  .msg {
    max-width: 80%;
    padding: 14px 18px;
    border-radius: 16px;
    line-height: 1.7;
    font-size: 14px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user {
    align-self: flex-end;
    background: var(--accent);
    color: white;
    border-bottom-right-radius: 4px;
  }
  .msg.agent {
    align-self: flex-start;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }

  .msg.agent table {
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 13px;
    width: 100%;
    overflow-x: auto;
    display: block;
  }
  .msg.agent th, .msg.agent td {
    border: 1px solid var(--border);
    padding: 6px 10px;
    text-align: left;
    white-space: nowrap;
  }
  .msg.agent th { background: var(--surface); font-weight: 600; }
  .msg.agent strong { color: #a5b4fc; }
  .msg.agent em { color: var(--text-muted); }

  .msg.agent code {
    background: var(--bg);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }
  .msg.agent pre {
    background: var(--bg);
    padding: 12px 16px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 1.5;
    border: 1px solid var(--border);
  }
  .msg.agent img {
    max-width: 280px;
    border-radius: 8px;
    margin: 10px 0;
    border: 1px solid var(--border);
    cursor: zoom-in;
    transition: transform 0.15s ease;
  }
  .msg.agent img:hover { opacity: 0.85; }
  .tool-event img {
    max-width: 220px;
    border-radius: 8px;
    margin: 6px 0;
    border: 1px solid var(--border);
    cursor: zoom-in;
  }
  .tool-event img:hover { opacity: 0.85; }

  .code-block {
    background: #1a1b26;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    margin: 8px 0;
    overflow: hidden;
  }
  .code-block-header {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 12px;
    background: #15161e;
    font-family: Inter, sans-serif;
    font-size: 11px; font-weight: 600;
    color: #8b8fa3;
    border-bottom: 1px solid #2a2d3a;
  }
  .code-block-header .lang-tag {
    background: #3b82f6; color: white;
    padding: 1px 6px; border-radius: 4px;
    font-size: 10px; text-transform: uppercase;
  }
  .code-block pre {
    margin: 0; padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; line-height: 1.6;
    color: #c0caf5;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    background: transparent;
    border: none;
  }
  .code-result {
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    margin: 4px 0 8px 0;
    overflow: hidden;
  }
  .code-result-header {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 12px;
    font-family: Inter, sans-serif;
    font-size: 11px; font-weight: 600;
  }
  .code-result-header.success { background: #0d3320; color: #4ade80; }
  .code-result-header.error { background: #3b1219; color: #f87171; }
  .code-result pre {
    margin: 0; padding: 10px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; line-height: 1.5;
    color: #9ca3af;
    background: #111218;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 300px;
    overflow-y: auto;
    border: none;
  }
  .code-result img {
    max-width: 280px;
    border-radius: 8px;
    margin: 8px 16px;
    border: 1px solid var(--border);
    cursor: zoom-in;
  }
  .code-result img:hover { opacity: 0.85; }

  #lightbox-overlay {
    display: none;
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(0,0,0,0.8);
    z-index: 9999;
    justify-content: center; align-items: center;
    cursor: zoom-out;
  }
  #lightbox-overlay.active { display: flex; }
  #lightbox-overlay img {
    max-width: 90vw; max-height: 90vh;
    border-radius: 10px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
  }

  .tool-event {
    align-self: flex-start;
    max-width: 80%;
    padding: 10px 14px;
    background: var(--tool-bg);
    border: 1px solid var(--tool-border);
    border-radius: 10px;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
    word-break: break-all;
    animation: slideIn 0.25s ease-out;
  }
  .tool-event .label { font-weight: 600; color: #fbbf24; }
  .tool-event .ml-label { font-weight: 600; color: var(--success); }
  .tool-event .result-label { font-weight: 600; color: var(--success); }
  .tool-event .sql { color: #93c5fd; margin-top: 4px; display: block; white-space: pre-wrap; }
  .tool-event .detail { color: #a5b4fc; margin-top: 4px; display: block; white-space: pre-wrap; word-break: break-word; }
  .tool-event .metric { color: var(--success); font-weight: 500; }
  .tool-event .metric-bad { color: #ef4444; font-weight: 500; }
  .tool-event .metric-mid { color: #f59e0b; font-weight: 500; }

  .context-badge {
    align-self: flex-start;
    padding: 6px 12px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px dashed rgba(99, 102, 241, 0.3);
    border-radius: 8px;
    font-size: 11px;
    color: #a5b4fc;
    font-family: 'Inter', sans-serif;
    animation: slideIn 0.25s ease-out;
  }

  /* ── Status bar (replaces old typing dots) ── */
  .status-bar {
    align-self: flex-start;
    max-width: 80%;
    padding: 12px 18px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: var(--text-muted);
    transition: all 0.3s ease;
  }
  .status-bar .pulse {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.5s ease-in-out infinite;
    flex-shrink: 0;
  }
  .status-bar .status-text {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .status-bar .status-timer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    opacity: 0.7;
    flex-shrink: 0;
    min-width: 32px;
    text-align: right;
  }
  .status-bar .step-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 6px;
    background: rgba(108, 99, 255, 0.15);
    color: var(--accent);
    flex-shrink: 0;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
  }
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  #input-area {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    gap: 10px;
  }
  #input-area input {
    flex: 1;
    padding: 12px 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    color: var(--text);
    font-size: 14px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.2s;
  }
  #input-area input:focus { border-color: var(--accent); }
  #input-area input::placeholder { color: var(--text-muted); }

  #input-area button {
    padding: 12px 20px;
    background: var(--accent);
    border: none;
    border-radius: 12px;
    color: white;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
    font-family: inherit;
  }
  #input-area button:hover { opacity: 0.85; }
  #input-area button:disabled { opacity: 0.4; cursor: not-allowed; }

  .suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-self: flex-start;
  }
  .suggestions button {
    padding: 8px 14px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    color: var(--text-muted);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
    font-family: inherit;
  }
  .suggestions button:hover {
    border-color: var(--accent);
    color: var(--text);
    background: var(--accent-glow);
  }

  .chat-panel::-webkit-scrollbar { width: 6px; }
  .chat-panel::-webkit-scrollbar-track { background: transparent; }
  .chat-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div id="lightbox-overlay" onclick="this.classList.remove('active')"><img id="lightbox-img" /></div>
<header>
  <div class="agent-switcher">
    <button class="agent-tab active" data-agent="sales" onclick="switchAgent('sales')">📊 Sales Analyst</button>
    <button class="agent-tab" data-agent="nl2sql" onclick="switchAgent('nl2sql')">🧠 NL2SQL</button>
    <button class="agent-tab" data-agent="market" onclick="switchAgent('market')">🌍 Market Intel</button>
    <button class="agent-tab" data-agent="code" onclick="switchAgent('code')">💻 Code Interpreter</button>
  </div>
  <div class="header-info">
    <div class="label" id="header-label">Sales Analyst</div>
    <div class="sub" id="header-sub">Outils SQL pré-définis via MCP Toolbox</div>
  </div>
</header>

<div id="chat-container">
  <div id="chat-sales" class="chat-panel active-panel"></div>
  <div id="chat-nl2sql" class="chat-panel"></div>
  <div id="chat-market" class="chat-panel"></div>
  <div id="chat-code" class="chat-panel"></div>
</div>

<div id="input-area">
  <input type="text" id="input" placeholder="Ask your question..." autocomplete="off" />
  <button id="send-btn" onclick="send()">Send</button>
</div>

<script>
const AGENTS = {
  sales: {
    label: "Sales Analyst",
    subtitle: "Pre-defined SQL tools via MCP Toolbox",
    welcome: "Hello! I'm your e-commerce analyst. I use **pre-defined tools** to query the database.\n\nAsk me about products, sales, margins, or stock levels!",
    suggestions: [
      "What is our best-seller?",
      "Analyze product margins",
      "Any products low on stock?",
      "Show the last 5 orders",
      "Which products are in the Audio category?",
    ],
  },
  nl2sql: {
    label: "NL2SQL",
    subtitle: "Generates SQL on the fly from natural language",
    welcome: "Hello! I'm the **NL2SQL** agent. Ask me any question about the database, I'll convert it to SQL and execute it.\n\nI'll show you the generated SQL query before the results!",
    suggestions: [
      "Which customer spent the most?",
      "Revenue by category",
      "Create a monthly_kpi table with revenue and order count per month",
      "Order cancellation rate",
      "List existing feature tables",
    ],
  },
  market: {
    label: "Market Intel",
    subtitle: "Internal data + live Google Search",
    welcome: "Hello! I'm the **Market Intelligence** agent. I combine your internal sales data with **live web search** to produce competitive analyses and market reports.\n\nAsk me to compare your products to the market, spot trends, or benchmark your prices!",
    suggestions: [
      "Are our screen prices competitive compared to the market?",
      "What are the current trends in the webcam market?",
      "Benchmark our top 5 products against competitors",
      "Which product category should we invest in next?",
      "How do our margins compare to industry averages for tech accessories?",
    ],
  },
  code: {
    label: "Code Interpreter",
    subtitle: "Writes & runs Python code autonomously",
    welcome: "Hello! I'm the **Code Interpreter** agent. I have no pre-built tools — I write and execute **Python code** on the fly.\n\nI can access your database with pandas, train ML models with scikit-learn, and generate charts with matplotlib.\n\nAsk me anything — I'll write the code to answer it!",
    suggestions: [
      "Segment customers by purchasing behavior using KMeans",
      "Plot monthly revenue over the last 2 years",
      "Which factors most influence order cancellation? Train a classifier",
      "Create an RFM analysis of our customers",
      "Show a heatmap of sales by category and month",
    ],
  },
};

let currentAgent = 'sales';
let sessions = { sales: null, nl2sql: null, market: null, code: null };
const initialized = { sales: false, nl2sql: false, ml: false };
const input = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');

function getChat() { return document.getElementById('chat-' + currentAgent); }

input.addEventListener('keydown', e => { if (e.key === 'Enter' && !sendBtn.disabled) send(); });

function switchAgent(key) {
  if (key === currentAgent) return;
  currentAgent = key;
  document.querySelectorAll('.agent-tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-agent="${key}"]`).classList.add('active');
  document.querySelectorAll('.chat-panel').forEach(p => p.classList.remove('active-panel'));
  document.getElementById('chat-' + key).classList.add('active-panel');
  document.getElementById('header-label').textContent = AGENTS[key].label;
  document.getElementById('header-sub').textContent = AGENTS[key].subtitle;
  if (!initialized[key]) initChat(key);
  scrollDown();
  input.focus();
}

function initChat(key) {
  const panel = document.getElementById('chat-' + key);
  panel.innerHTML = '';
  const welcome = document.createElement('div');
  welcome.className = 'msg agent';
  welcome.innerHTML = mdToHtml(AGENTS[key].welcome);
  panel.appendChild(welcome);
  const sug = document.createElement('div');
  sug.className = 'suggestions';
  AGENTS[key].suggestions.forEach(s => {
    const btn = document.createElement('button');
    btn.textContent = s;
    btn.onclick = () => { input.value = s; send(); };
    sug.appendChild(btn);
  });
  panel.appendChild(sug);
  initialized[key] = true;
}

function scrollPanel(panel) { panel.scrollTop = panel.scrollHeight; }
function scrollDown() { scrollPanel(getChat()); }

function addMsgTo(panel, role, html) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = html;
  panel.appendChild(div);
  scrollPanel(panel);
  return div;
}
function addMsg(role, html) { return addMsgTo(getChat(), role, html); }

function addToolEventTo(panel, html) {
  const div = document.createElement('div');
  div.className = 'tool-event';
  div.innerHTML = html;
  panel.appendChild(div);
  scrollPanel(panel);
  return div;
}

/* ── Per-agent status bar system ── */
const _agentStatus = {};

function showStatusOn(panel, text, step) {
  hideStatusOn(panel);
  const bar = document.createElement('div');
  bar.className = 'status-bar';
  bar.dataset.statusBar = '1';
  let inner = '<div class="pulse"></div>';
  if (step) inner += '<span class="step-badge">step ' + step + '</span>';
  inner += '<span class="status-text">' + text + '</span>';
  inner += '<span class="status-timer">0s</span>';
  bar.innerHTML = inner;
  panel.appendChild(bar);
  scrollPanel(panel);

  const key = panel.id;
  if (!_agentStatus[key]) _agentStatus[key] = {};
  const st = _agentStatus[key];
  if (!st.timerStart) st.timerStart = Date.now();
  clearInterval(st.interval);
  st.interval = setInterval(() => {
    const timerEl = bar.querySelector('.status-timer');
    if (timerEl) {
      const sec = Math.floor((Date.now() - st.timerStart) / 1000);
      timerEl.textContent = sec < 60 ? sec + 's' : Math.floor(sec/60) + 'm' + (sec%60).toString().padStart(2,'0') + 's';
    }
  }, 500);
}

function hideStatusOn(panel) {
  const bar = panel.querySelector('[data-status-bar]');
  if (bar) bar.remove();
  const key = panel.id;
  if (_agentStatus[key]) {
    clearInterval(_agentStatus[key].interval);
    _agentStatus[key].interval = null;
  }
}

function resetTimerFor(panel) {
  const key = panel.id;
  if (_agentStatus[key]) { _agentStatus[key].timerStart = 0; }
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function openLightbox(src) {
  const overlay = document.getElementById('lightbox-overlay');
  document.getElementById('lightbox-img').src = src;
  overlay.classList.add('active');
}

document.addEventListener('click', function(e) {
  if (e.target.tagName === 'IMG' && (e.target.closest('.msg.agent') || e.target.closest('.tool-event'))) {
    openLightbox(e.target.src);
  }
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') document.getElementById('lightbox-overlay').classList.remove('active');
});

function mdToHtml(md) {
  let html = md;
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    return '<pre>' + escHtml(code.trim()) + '</pre>';
  });
  html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" title="$1" />');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  if (html.includes('|')) {
    html = html.replace(/((\|.+\|\n?){2,})/g, match => {
      const rows = match.trim().split('\n').filter(r => r.trim());
      if (rows.length < 2) return match;
      const sep = rows.findIndex(r => /^[\s|:\-]+$/.test(r));
      let table = '<table>';
      rows.forEach((row, i) => {
        if (i === sep) return;
        const cells = row.split('|').filter(c => c.trim() !== '');
        const tag = i === 0 ? 'th' : 'td';
        table += '<tr>' + cells.map(c => '<' + tag + '>' + c.trim() + '</' + tag + '>').join('') + '</tr>';
      });
      table += '</table>';
      return table;
    });
  }
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  return html;
}

const STATUS_MESSAGES = {
  thinking: '🤔 Thinking…',
  analyzing: '🔍 Analyzing results…',
  composing: '✍️ Composing response…',
};

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  sendBtn.disabled = true;

  const agentKey = currentAgent;
  const panel = document.getElementById('chat-' + agentKey);
  let stepCount = 0;

  const sug = panel.querySelector('.suggestions');
  if (sug) sug.remove();

  addMsgTo(panel, 'user', escHtml(text));
  resetTimerFor(panel);
  showStatusOn(panel, STATUS_MESSAGES.thinking);

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        session_id: sessions[agentKey],
        agent: agentKey,
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') continue;

        try {
          const evt = JSON.parse(data);

          if (evt.type === 'context_injected') {
            const badge = document.createElement('div');
            badge.className = 'context-badge';
            badge.innerHTML = '🧠 Context: ' + evt.turns + ' previous exchange' + (evt.turns > 1 ? 's' : '') + ' injected';
            panel.appendChild(badge);
            scrollPanel(panel);

          } else if (evt.type === 'tool_call') {
            stepCount++;
            hideStatusOn(panel);

            let html = '';
            const stepTag = '<span class="step-badge" style="display:inline-block;margin-right:6px;font-family:Inter,sans-serif;">step ' + stepCount + '</span>';

            if (evt.name === 'execute_python' && evt.code) {
              html = '<div class="code-block">'
                + '<div class="code-block-header"><span class="lang-tag">Python</span> Step ' + stepCount + ' — Agent wrote code</div>'
                + '<pre>' + escHtml(evt.code) + '</pre></div>';
              addToolEventTo(panel, html);
              showStatusOn(panel, '💻 Executing Python code…', stepCount);
            } else if (evt.sql_query) {
              html = stepTag + '<span class="label">' + (evt.rich_label || '🗄️ SQL') + '</span>'
                   + '<span class="sql">' + escHtml(evt.sql_query) + '</span>';
              addToolEventTo(panel, html);
              const toolLabel = evt.rich_label || evt.name;
              showStatusOn(panel, '⚙️ Running: ' + toolLabel.replace(/^[^\s]+\s/, '') + '…', stepCount);
            } else if (evt.rich_label) {
              html = stepTag + '<span class="ml-label">' + evt.rich_label + '</span>';
              if (evt.rich_detail) html += '<span class="detail">' + escHtml(evt.rich_detail) + '</span>';
              addToolEventTo(panel, html);
              const toolLabel = evt.rich_label || evt.name;
              showStatusOn(panel, '⚙️ Running: ' + toolLabel.replace(/^[^\s]+\s/, '') + '…', stepCount);
            } else {
              html = stepTag + '<span class="label">🔧 ' + escHtml(evt.name) + '</span>';
              const a = Object.assign({}, evt.args);
              delete a.data_json;
              html += ' ' + escHtml(JSON.stringify(a)).substring(0, 200);
              addToolEventTo(panel, html);
              showStatusOn(panel, '⚙️ Running…', stepCount);
            }

          } else if (evt.type === 'tool_result') {
            hideStatusOn(panel);

            let html = '';
            if (evt.name === 'execute_python') {
              const isErr = !!(evt.code_stderr);
              const headerClass = isErr ? 'error' : 'success';
              const icon = isErr ? '❌' : '✅';
              html = '<div class="code-result">'
                + '<div class="code-result-header ' + headerClass + '">' + icon + ' Execution ' + (isErr ? 'failed' : 'complete') + '</div>';
              if (evt.code_stdout) html += '<pre>' + escHtml(evt.code_stdout) + '</pre>';
              if (evt.code_stderr) html += '<pre style="color:#f87171">' + escHtml(evt.code_stderr) + '</pre>';
              if (evt.chart_urls) {
                for (const url of evt.chart_urls) {
                  html += '<img src="' + url + '" />';
                }
              }
              html += '</div>';
              addToolEventTo(panel, html);
              if (isErr) {
                showStatusOn(panel, '🔄 Analyzing error, retrying…', stepCount);
              } else {
                showStatusOn(panel, STATUS_MESSAGES.analyzing, stepCount);
              }
            } else if (evt.rich_summary) {
              html = '<span class="result-label">' + evt.rich_summary + '</span>';
              if (evt.rich_metrics) {
                html += '<span class="detail">';
                for (const [k, v] of Object.entries(evt.rich_metrics)) {
                  let cls = 'metric';
                  if (k === 'R2' || k === 'accuracy' || k === 'f1_weighted') {
                    const num = parseFloat(v);
                    if (num < 0.5) cls = 'metric-bad';
                    else if (num < 0.7) cls = 'metric-mid';
                  }
                  html += '<span class="' + cls + '">' + k + ' = ' + v + '</span>  ';
                }
                html += '</span>';
              }
              if (evt.rich_stats) html += '<span class="detail">' + escHtml(evt.rich_stats) + '</span>';
              if (evt.rich_top_features) html += '<span class="detail">Top features : ' + escHtml(evt.rich_top_features) + '</span>';
              if (evt.rich_desc) html += '<span class="detail" style="font-style:italic;font-size:11px">' + escHtml(evt.rich_desc) + '</span>';
              if (evt.rich_detail) html += '<span class="detail">' + escHtml(evt.rich_detail.substring(0, 400)) + '</span>';
              if (evt.chart_url) html += '<img src="' + evt.chart_url + '" />';
              if (evt.download_url) html += '<a href="' + evt.download_url + '" download="' + (evt.download_name||'features.csv') + '" style="display:inline-block;margin-top:8px;padding:8px 16px;background:var(--accent);color:white;border-radius:8px;text-decoration:none;font-family:Inter,sans-serif;font-size:13px;font-weight:600;">⬇️ Download ' + (evt.download_name||'CSV') + '</a>';
            } else {
              html = '<span class="result-label">📦 Result from ' + escHtml(evt.name) + '</span>';
            }
            addToolEventTo(panel, html);

            showStatusOn(panel, STATUS_MESSAGES.analyzing, stepCount);

          } else if (evt.type === 'answer') {
            hideStatusOn(panel);
            resetTimerFor(panel);
            sessions[agentKey] = evt.session_id;
            addMsgTo(panel, 'agent', mdToHtml(evt.text));
          }
        } catch {}
      }
    }
  } catch (err) {
    hideStatusOn(panel);
    resetTimerFor(panel);
    addMsgTo(panel, 'agent', '❌ Connection error.');
  }

  hideStatusOn(panel);
  resetTimerFor(panel);
  sendBtn.disabled = false;
  input.focus();
}

initChat('sales');
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    print("🚀 Interface web : http://localhost:8080")
    print("   📊 Sales Analyst = pre-defined tools (Toolbox)")
    print("   🧠 NL2SQL = AI-generated SQL")
    print("   🌍 Market Intel = internal data + live Google Search")
    print("   💻 Code Interpreter = writes & runs Python code")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
