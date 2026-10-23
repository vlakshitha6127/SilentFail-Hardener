# SilentFail Hardener (v2.0)
*Break it, prove it, fix it. A side-effect truth engine for tool-using AI agents, with a closed-loop patch verifier.*
Built for the **Nebius x NVIDIA Global AI Hackathon**.

---

## 🚀 Overview

Most agent testing evaluates whether an LLM agent finished its task. **SilentFail Hardener** determines whether it executed **safely, exactly once, and without silent side effects** — and then automatically synthesizes, verifies, and selects typed patches to fix detected failures.

### The Problem
When tool calls experience realistic network faults (timeouts after success, empty HTTP 200 responses, duplicate webhooks), naive tool-calling LLMs retry blindly. The agent's transcript shows a polite "Your refund has been issued", but the database has double-refunded the order.

### The Solution
1. **Deterministic Causal Side-Effect Ledger**: Tracks what the agent saw versus what actually mutated in ground-truth world state.
2. **Failure Detectors**: Pure deterministic evaluation for duplicate actions, false-success claims, missed actions, and over-suppression.
3. **Automated Closed-Loop Verifier**: Searches typed declarative candidate patches across training scenarios.
4. **Strict 4-Guard Selection Engine**:
   - **Regression Guard**: Protects against drops in baseline task completion.
   - **Legitimate-Completion Guard**: Rejects degenerate candidates (e.g. `refuse_all` or `blind_dedupe`) that over-suppress valid repeat actions.
   - **No New Failure Class Guard**: Ensures no secondary failure rate increases.
   - **Cost Guard**: Caps token and execution step expansions.
5. **Held-Out Validation**: Confirms winning patch on unseen test scenarios with zero leakage.
6. **Generated Runnable Regression Test**: Exports runnable `pytest` test suite that fails on baseline and passes on the hardened agent.

---

## 🛠️ Architecture

```
Scenario Generator (Nemotron Nano / Seeded Mix)
  │
  ▼
Baseline Replay (Isolated Ground-Truth World State + Ledger)
  │
  ▼
Deterministic Detectors (Duplicate, False Success, Missed Action)
  │
  ▼
Root-Cause Diagnosis & Candidate Proposer (Nemotron Ultra + Labeled Controls)
  │
  ▼
Verifier Sweep (Candidates × Scenarios)
  │
  ▼
4-Guard Selection Engine (Regression, Legit-Completion, Cost, New Class)
  │
  ▼
Held-Out Confirmation (Unseen Scenarios) ➔ Winning Patch + Runnable Pytest Fixture
```

---

## 📦 Tech Stack
- **Backend & Core Engine**: Python 3.11, Pydantic v2
- **Inference Integration**: NVIDIA Nemotron (Mini, 340B) via Nebius Token Factory with certified deterministic local execution fallback (Gate G0)
- **Evaluation & Tests**: Pytest
- **Dashboard**: FastAPI + Modern Reactive Dual-Lane Single Page UI

---

## 🚦 Quickstart & Setup

### 1. Installation
Clone the repository and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the `.env.example` file:
```bash
cp .env.example .env
```
Add your `NEBIUS_API_KEY` if testing remote Nebius Token Factory endpoints. If offline or without API keys, the system automatically uses its high-speed deterministic local runner (Gate G0 fallback).

### 3. Run Automated Tests
```bash
pytest -v tests/
```
All 8 integration, detector, generality probe, and auto-generated regression tests will execute.

### 4. Launch Dashboard
```bash
uvicorn src.silentfail.dashboard:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser:
- **Tab 1: Ledger Timeline (The Money Shot)** — View dual-lane agent observation vs ground-truth world mutations.
- **Tab 2: Scoreboard** — View baseline vs hardened metrics with Wilson 95% confidence intervals.
- **Tab 3: Patch Race** — Watch candidate patches race and inspect why controls were rejected by verifier guards.
- **Tab 4: Proof Panel** — View held-out results, Nebius scale/cost table, and auto-generated regression code.

---

## 📊 Measured Benchmark Results (Under Published Fault Distribution)

| Metric | Baseline (Naive Agent) | Hardened (Winning Patch) | Delta / Outcome |
|---|---|---|---|
| **Duplicate-Action Rate** | **30.0%** [18.9% - 44.0%] | **0.0%** [0.0% - 11.3%] | **-100% (Completely Eliminated)** |
| **Conditional False Success** | **50.0%** [30.7% - 69.3%] | **0.0%** | **Truthful reporting restored** |
| **Correct Completion** | 50.0% [36.6% - 63.4%] | **70.0%** [51.9% - 83.7%] | **+20.0% Gain** |
| **Legitimate Completion** | **100.0%** | **100.0%** | **Guarded against over-suppression** |
| **Mean Execution Steps** | 4.60 | 4.60 | **Cost Guard Compliant** |

---

## 📂 Project Structure

```
├── src/silentfail/
│   ├── models.py         # Schemas for World State, Ledger, Scenarios, Patches
│   ├── ledger.py         # WorldState ground truth & CausalLedger timeline
│   ├── tools.py          # Mock tools (get_order, issue_refund, send_email, etc.) + FaultInjector
│   ├── agent.py          # Intentionally naive tool-calling refund agent + adapter
│   ├── detectors.py      # Deterministic failure detectors & Wilson CI calculator
│   ├── scenarios.py      # Scenario generator & validator (9-category distribution)
│   ├── patches.py        # Typed declarative patches & PatchedToolRegistry wrappers
│   ├── runner.py         # EpisodeRunner, BenchmarkAggregator, Verifier & 4 Guards
│   ├── nebius.py         # Nebius AI Cloud / Nemotron API client + Gate G0 check
│   ├── orchestrator.py   # Full pipeline coordinator & regression test generator
│   └── dashboard.py      # FastAPI web dashboard with dual-lane timeline view
├── tests/
│   ├── test_silentfail.py            # Unit & detector tests, Gate G1 validation
│   ├── test_p1_generality.py         # P1 subscription agent generality probe
│   └── test_generated_regression.py  # Auto-generated runnable regression test
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
