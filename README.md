# Healthcare Support & Care Navigation Agent

A healthcare navigation prototype powered by **LangGraph** orchestration,
a **Neo4j** Knowledge Graph, **SQLite**, and **LLM function calling**.
---

## What It Does

Users send natural language healthcare queries to a single `/chat` endpoint.
The LangGraph agent autonomously decides which tool to call, executes it,
and returns a synthesised answer.

**Example queries:**
- *"Which doctors treat hypertension?"*
- *"What is the status of appointment AP101?"*
- *"My weight is 72 kg and height is 175 cm. What is my BMI?"*
- *"I have a migraine. Who should I see?"*
- *"What is diabetes?"* ← answered directly, no tool call needed

---

## Architecture

```
User Request (POST /chat)
        │
        ▼
   FastAPI Layer
   (validates request, calls run_agent via asyncio.to_thread)
        │
        ▼
  ┌─────────────────────────────┐
  │     LangGraph StateGraph    │
  │                             │
  │  [START] → agent_node       │
  │               │             │
  │         should_continue?    │
  │         /            \      │
  │      "tools"         END    │
  │         │             │     │
  │     tool_node         │     │
  │         │             │     │
  │     agent_node ◄──────┘     │
  │         │                   │
  │        END                  │
  └─────────────────────────────┘
        │
        ▼
   FastAPI Response
   { answer, total_turns }
```

### LangGraph Nodes

| Node | Responsibility |
|---|---|
| `agent_node` | Calls the LLM with full message history and bound tools. Returns AIMessage. |
| `tool_node_with_tracking` | Wraps LangGraph's built-in ToolNode. Executes tool calls, records them in state. |

### Conditional Routing (`should_continue`)

After `agent_node` runs, the router checks the last message:

- `total_turns >= MAX_AGENT_TURNS` → **END** (iteration limit)
- `state.error is set` → **END** (LLM failure)
- Last message has `tool_calls` → **"tools"**
- Last message has no `tool_calls` → **END** (final answer)

### Graph State (`AgentState`)

```python
class AgentState(TypedDict):
    messages:     list[BaseMessage]   # full conversation history
    tools_used:   list[dict]          # [{tool, args}, ...] for observability
    total_turns:  int                 # tool round-trips completed
    request_id:   str                 # UUID for log correlation
    error:        str | None          # set on graph-level failure
```

`messages` uses LangGraph's `add_messages` reducer — new messages are
appended, not replaced.

---

## Tools

### 1. `find_doctor`
- **Backend:** Neo4j Knowledge Graph
- **Search modes:** by medical condition OR by specialization
- **Multi-hop:** Condition → TREATED_BY → Specialization ← SPECIALIZES_IN ← Doctor → WORKS_AT → Hospital
- **Extras:** Returns related conditions for richer context

### 2. `get_appointment`
- **Backend:** SQLite
- **Input:** Appointment ID (format: AP + digits, e.g. `AP101`)
- **Output:** Patient name, doctor, date, time, status, notes
- **Normalisation:** Case-insensitive; strips whitespace

### 3. `calculate_bmi`
- **Backend:** Pure Python
- **Formula:** `BMI = weight_kg / (height_m²)`
- **Categories:** Underweight / Normal weight / Overweight / Obese (WHO)
- **Not a diagnosis:** Result includes an explicit disclaimer

---

## Neo4j Knowledge Graph Model

```
(:Doctor {name, experience_years})
    -[:SPECIALIZES_IN]→ (:Specialization {name})
    -[:WORKS_AT]→       (:Hospital {name, location})

(:Condition {name})
    -[:TREATED_BY]→   (:Specialization {name})
    -[:RELATED_TO]→   (:Condition {name})
```

### Seeded Data
- **6 doctors** (Dr. Sharma, Dr. Patel, Dr. Mehta, Dr. Rao, Dr. Gupta, Dr. Krishnarao)
- **4 hospitals** (City Hospital, Apollo Clinic, Sunrise Medical, Global Health Hub)
- **5 specializations** (Cardiology, Neurology, Orthopedics, General Medicine, Endocrinology)
- **8 conditions** (Hypertension, Heart Disease, Migraine, Knee Pain, Diabetes, Thyroid Disorder, Arthritis, Stroke)

### Note on Case-Insensitive Queries
Queries use `toLower()` for case-insensitive matching. This prevents Neo4j
from using standard property indexes. For a prototype with small data this is
acceptable. In production, use a full-text index instead:
```cypher
CREATE FULLTEXT INDEX condition_name FOR (c:Condition) ON EACH [c.name]
```

---

## Project Structure

```
healthcare-agent/
├── app/
│   ├── agent.py            # LangGraph graph definition (core)
│   ├── config.py           # Centralised environment config
│   ├── logger.py           # Structured logging setup
│   ├── main.py             # FastAPI app, lifespan, endpoints
│   ├── schemas.py          # Pydantic request/response models
│   ├── graph/
│   │   └── neo4j_client.py # Neo4j driver + Cypher queries
│   └── tools/
│       ├── calculate_bmi.py
│       ├── find_doctor.py
│       └── get_appointment.py
├── data/
│   └── seed_graph.py       # Neo4j seed script
├── tests/
│   ├── test_calculate_bmi.py
│   ├── test_find_doctor.py
│   ├── test_get_appointment.py
│   ├── test_agent.py
│   └── test_api.py
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- Neo4j 5.x (Community Edition is sufficient)
- An [OpenRouter](https://openrouter.ai) API key

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   OPENROUTER_API_KEY=your_key_here
#   NEO4J_PASSWORD=your_password
```

### 3. Start Neo4j

Start Neo4j locally (Community Edition). Default bolt port is 7687.
Create a database and set the password to match your `.env`.

### 4. Seed the Knowledge Graph

```bash
python data/seed_graph.py
```

Expected output:
```
Connecting to Neo4j at bolt://localhost:7687...
Seeding Neo4j Knowledge Graph...

  Created 4 uniqueness constraints.
  Cleared existing graph data.
  Created 5 Specialization nodes.
  Created 4 Hospital nodes.
  Created 6 Doctor nodes.
  Created 8 Condition nodes.
  Created 6 SPECIALIZES_IN relationships.
  Created 6 WORKS_AT relationships.
  Created 8 TREATED_BY relationships.
  Created 6 RELATED_TO relationships.

Graph seeded successfully.
```

### 5. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Test the API

```bash
# Health check
curl http://localhost:8000/

# Extended health (checks Neo4j)
curl http://localhost:8000/health

# Doctor search by condition
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Which doctors treat hypertension?"}'

# Appointment lookup
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the status of appointment AP101?"}'

# BMI calculation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "My weight is 72 kg and height is 175 cm. What is my BMI?"}'

# Direct answer (no tool call)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is hypertension?"}'
```

Swagger UI is available at: **http://localhost:8000/docs**

---

## Running Tests

```bash
# All tests (Neo4j and LLM are mocked — no real services needed)
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/test_calculate_bmi.py -v
pytest tests/test_agent.py -v
```

---

## Error Handling

| Error scenario | Behaviour |
|---|---|
| Invalid request body | FastAPI returns 422 with validation details |
| Missing appointment ID | Tool returns `found: False` with a clear message |
| Condition not in graph | Tool returns empty list with a message |
| Neo4j unavailable | Tool returns safe error dict; raw error is logged, not returned |
| LLM API failure | Agent returns fallback message; raw error is logged |
| Max iterations reached | Agent stops and returns current answer |
| Unknown tool name | ToolNode returns error message to LLM; LLM reports to user |

---

## API Reference

### `POST /chat`
**Request:**
```json
{ "message": "string (1–1000 chars)" }
```
**Response:**
```json
{
  "answer": "string",
  "tools_used": [{ "tool": "string", "args": {} }],
  "total_turns": 0
}
```

### `GET /`
Basic liveness check. Returns `{ "status": "ok" }`.

### `GET /health`
Extended check including Neo4j connectivity.
Returns 200 if healthy, 503 if Neo4j is unreachable.