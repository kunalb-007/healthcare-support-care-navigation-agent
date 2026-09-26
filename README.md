# Healthcare Support & Care Navigation Agent

An **agentic AI healthcare navigation prototype** built with **LangGraph, LLM tool calling, Neo4j Knowledge Graph, SQLite, FastAPI, and Python**.

The agent accepts natural-language healthcare queries, autonomously selects the appropriate tool, executes it, and synthesizes the final response.


📖 **Swagger API Docs:**   
https://healthcare-support-care-navigation-agent.onrender.com/docs

---

## What It Does

| Capability                     | Technology                              |
| ------------------------------ | --------------------------------------- |
| Doctor / specialist discovery  | **Neo4j Knowledge Graph + Cypher**      |
| Appointment lookup             | **SQLite**                              |
| BMI calculation                | **Deterministic Python**                |
| Tool selection & orchestration | **LangGraph + LLM function calling**    |
| API                            | **FastAPI**                             |
| Observability                  | Structured logging + tool-call tracking |

### Example Queries

```text
"Which doctors treat hypertension?"
"What is the status of appointment AP101?"
"My weight is 72 kg and height is 175 cm. What is my BMI?"
"I have a migraine. Who should I see?"
"What is hypertension?"       → direct LLM response
```

---

## Architecture

```text
                    User
                     │
                 POST /chat
                     ▼
               ┌───────────┐
               │  FastAPI  │
               └─────┬─────┘
                     │
                     ▼
          ┌─────────────────────┐
          │    LangGraph        │
          │                     │
          │    Agent Node       │
          │        │            │
          │    tool call?       │
          │    /         \      │
          │  yes         no     │
          │   │          │      │
          │   ▼          ▼      │
          │ ToolNode     END    │
          │   │                 │
          │   └──► Agent        │
          └─────────┬───────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      Neo4j       SQLite      Python
        │           │           │
     Doctors    Appointments    BMI
```

### Agent Loop

```text
User Query
    ↓
LLM decides whether a tool is required
    ↓
LangGraph executes tool
    ↓
Tool result returned to LLM
    ↓
LLM synthesizes final response
```

The graph is **stateless per request** and supports a configurable maximum number of agent/tool turns.

---

## Tools

### `find_doctor`

Uses Neo4j to traverse:

```text
Condition
   ↓ TREATED_BY
Specialization
   ↑ SPECIALIZES_IN
Doctor
   ↓ WORKS_AT
Hospital
```

Supports searches by **condition or specialization** and returns doctor, specialization, hospital, location, and experience information.

### `get_appointment`

Queries SQLite using an appointment ID such as:

```text
AP101
```

Returns appointment, patient, doctor, date/time, status, and notes.

### `calculate_bmi`

Deterministic Python calculation:

```text
BMI = weight_kg / height_m²
```

Includes BMI category and informational disclaimer.

---

## Knowledge Graph

```text
(:Condition)
    ├──[:TREATED_BY]──► (:Specialization)
    │                         ▲
    │                         │
    │                  [:SPECIALIZES_IN]
    │                         │
    │                      (:Doctor)
    │                         │
    │                    [:WORKS_AT]
    │                         ▼
    │                     (:Hospital)
    │
    └──[:RELATED_TO]──► (:Condition)
```

Synthetic seed data:

```text
6 Doctors
4 Hospitals
5 Specializations
8 Conditions
26 Relationships
```

---

## Tech Stack

**Python · FastAPI · LangGraph · OpenRouter · Neo4j · Cypher · SQLite · Pydantic · Pytest**

---

## Run Locally

### 1. Install

```bash
python -m venv venv
venv\Scripts\activate        
pip install -r requirements.txt
```

### 2. Configure

Create `.env`:

```env
OPENROUTER_API_KEY=your_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Seed Neo4j

```bash
python data/seed_graph.py
```

### 4. Start API

```bash
uvicorn app.main:app --reload --port 8000
```

Swagger UI:
```text
http://localhost:8000/docs
```

### 5. Run Tests

```bash
pytest
```

---

## API

### `POST /chat`

```json
{
  "message": "Which doctors treat hypertension?"
}
```

Response:

```json
{
  "answer": "...",
  "tools_used": [
    {
      "tool": "find_doctor",
      "args": {
        "condition": "Hypertension"
      }
    }
  ],
  "total_turns": 1
}
```

### Health Endpoints

```text
GET /          → liveness
GET /health    → Neo4j connectivity check
```

---
