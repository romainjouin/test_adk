# ADK Course — Multi-Agent Sales Analytics Platform

## Duration: 15 minutes

---

## The Story (for the instructor)

> **"The data scientist who talked to their database"**
>
> You're a data scientist in a tech e-commerce startup.
> Every morning, the product manager shows up with questions:
>
> *"What's our best-seller? Any stockouts? Recent cancellations?"*
>
> You open Jupyter, write 3 SQL queries, format the results, paste them in Slack...
>
> One morning, you think:
>
> *"What if the PM could ask questions in plain language
>  directly to the database?"*
>
> Not by giving them SQL access (danger!), but by building an **AI agent**
> that knows which tools to use to query the database **securely**.
>
> And then you go further: what if we had **4 agents**, each with
> a different level of power?
>
> That's exactly what we build with **Google ADK** — from pre-defined
> secure tools to an agent that writes its own Python code.

---

## Architecture — 4 Agents, 4 Levels of Power

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Web UI (FastAPI + SSE)                            │
│                      http://localhost:8080                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │  Sales   │  │  NL2SQL  │  │  Market  │  │  Code Interpreter   │  │
│  │ Analyst  │  │          │  │  Intel   │  │                     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┬──────────┘  │
└───────┼──────────────┼─────────────┼───────────────────┼─────────────┘
        │              │             │                   │
   Toolbox tools   Raw SQL      SQL + Google       execute_python
   (pre-defined)   (generated)  Search API       (any Python code)
        │              │             │                   │
   ┌────▼──────────────▼─────────────▼───────────────────▼──────┐
   │                        shop.db                             │
   │  products | customers | orders  (3 years of data)          │
   └────────────────────────────────────────────────────────────┘
```

### Pedagogical Escalation

| Tab | Agent | Capability | Risk Level |
|-----|-------|-----------|------------|
| 1 | **Sales Analyst** | Pre-defined SQL tools via MCP Toolbox | Secure, limited |
| 2 | **NL2SQL** | AI-generated SQL queries + feature engineering | Flexible SQL |
| 3 | **Market Intel** | Internal data + live Google Search + charts | Multi-source |
| 4 | **Code Interpreter** | Writes & runs arbitrary Python code | Ultimate flexibility |

Each tab increases capability AND risk — perfect for teaching trade-offs.

---

## The 4 Agents in Detail

### 1. Sales Analyst (`sales_agent/`)

The **secure** agent. Uses pre-defined SQL tools exposed via MCP Toolbox.

- **Tools**: 7 SQL queries defined in `tools.yaml` (get-sales-by-product, search-products-by-name, etc.)
- **Architecture**: Agent -> Toolbox HTTP server -> SQLite
- **Key concept**: The agent never writes SQL — it only calls pre-defined, parameterized queries
- **Delegation**: Can delegate complex questions to the NL2SQL agent (Agent-as-Tool pattern)

### 2. NL2SQL (`nl2sql_agent/`)

The **flexible SQL** agent. Converts natural language to SQL queries.

- **Tools**: `execute_sql` (runs any SELECT on shop.db) + feature engineering tools
- **Feature engineering**: Can create local SQLite databases, add columns, insert query results, export to CSV
- **Key concept**: More powerful than pre-defined tools, but SQL injection risk
- **Dynamic context**: Knows today's date, product catalog, and DB statistics

### 3. Market Intelligence (`market_agent/`)

The **multi-source** agent. Combines internal data with live web search.

- **Tools**: `execute_sql`, `GoogleSearchAgentTool` (ADK sub-agent), `generate_comparison_chart`
- **Key concept**: Enriches internal data with external market data
- **Output**: Generates comparison charts (internal vs. market prices) saved as PNG

### 4. Code Interpreter (`code_agent/`)

The **ultimate flexibility** agent. Writes and executes arbitrary Python code.

- **Tools**: `execute_python` — runs any Python code locally with state persistence
- **Packages**: pandas, numpy, sklearn, matplotlib, sqlite3
- **Key concept**: No pre-built tools — the agent creates its own on the fly
- **Self-healing**: If code fails, reads the error, fixes it, and retries
- **Wow moment**: Ask "Segment customers by KMeans" and watch it write code, debug itself, and produce a scatter plot

---

## Key ADK Concepts Demonstrated

| Concept | Where |
|---------|-------|
| **Agent with tools** | Sales Analyst (Toolbox tools) |
| **MCP Toolbox** | `tools.yaml` — SQL as YAML-declared tools |
| **Agent-as-Tool** | Sales Analyst delegates to NL2SQL |
| **Dynamic instructions** | `db_context.py` injects product lists, dates |
| **Google Search sub-agent** | Market Intel uses `GoogleSearchAgentTool` |
| **Custom Python tools** | NL2SQL (feature engineering), Code Interpreter (`execute_python`) |
| **Conversation memory** | Backend tracks exchanges for context continuity |

---

## Quick Start

```bash
# 1. Virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Gemini API key → https://aistudio.google.com/app/apikey
#    Copy to each agent directory:
echo 'GOOGLE_API_KEY="your-key"' > sales_agent/.env
cp sales_agent/.env nl2sql_agent/.env
cp sales_agent/.env market_agent/.env
cp sales_agent/.env code_agent/.env

# 4. Create the database (3 years of realistic e-commerce data)
python setup_db.py

# 5. Download MCP Toolbox (for the Sales Analyst agent)
#    See https://mcp-toolbox.dev/ — place the binary as ./toolbox

# 6. Start the Toolbox server (in a separate terminal)
./toolbox --config tools.yaml --port 5050

# 7. Start the web app
python web_app.py
#    → Open http://localhost:8080
```

---

## Project Structure

```
test_adk/
├── README.md              ← This file
├── requirements.txt       ← Python dependencies
├── setup_db.py            ← Creates shop.db with 3 years of data
├── db_context.py          ← Dynamic context injection (dates, products)
├── tools.yaml             ← MCP Toolbox config (7 SQL tools)
├── web_app.py             ← FastAPI web UI with SSE streaming
├── run_demo.py            ← CLI demo script (standalone)
│
├── sales_agent/           ← Agent 1: Pre-defined tools (secure)
│   ├── agent.py
│   └── .env               ← API key (not tracked)
│
├── nl2sql_agent/          ← Agent 2: AI-generated SQL (flexible)
│   ├── agent.py
│   └── .env
│
├── market_agent/          ← Agent 3: DB + Google Search (multi-source)
│   ├── agent.py
│   └── .env
│
├── code_agent/            ← Agent 4: Python code execution (ultimate)
│   ├── agent.py
│   └── .env
│
├── ml_agent/              ← Legacy ML agent (replaced by Code Interpreter)
│   └── agent.py
│
├── static/
│   ├── charts/            ← Generated chart PNGs (not tracked)
│   └── exports/           ← Generated CSV exports (not tracked)
│
├── shop.db                ← SQLite database (generated, not tracked)
├── toolbox                ← MCP Toolbox binary (downloaded, not tracked)
└── features/              ← Feature engineering DBs (generated, not tracked)
```

---

## Suggested Demo Flow (15 min)

| Time | What to show |
|------|-------------|
| 0-2 min | The story + architecture diagram (4 agents, 4 levels) |
| 2-4 min | Open `tools.yaml` — show pre-defined SQL tools |
| 4-6 min | Open `sales_agent/agent.py` — show ToolboxToolset + Agent |
| 6-8 min | **Live demo**: Sales Analyst tab — "What's our best-seller?" |
| 8-10 min | Switch to NL2SQL tab — "Show revenue by month for 2024" |
| 10-12 min | Switch to Code Interpreter — "Segment customers with KMeans" |
| 12-13 min | Watch it write code, hit an error, self-correct, produce a chart |
| 13-15 min | Recap the 4 levels of capability vs. risk + Q&A |

---

## Going Further

- Replace SQLite with **PostgreSQL** or **BigQuery** (change the `source` in tools.yaml)
- Add **write tools** (UPDATE, INSERT) for order management
- Add **authentication** to restrict data per user
- Deploy with **Cloud Run** or **Vertex AI Agent Builder**
- Docs: https://google.github.io/adk-docs/ and https://mcp-toolbox.dev/
