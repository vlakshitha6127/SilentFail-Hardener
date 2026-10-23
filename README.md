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
4. **Guarded Selection Engine**:
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
Guarded Selection Engine (Regression, Legit-Completion, Cost, New Class)
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
py -m pytest -q
```
On Linux / macOS:
```bash
python -m pytest -q
```

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

## 📊 Benchmark Results (Traced to `data/latest_pipeline_run.json`)

> [!NOTE]
> **Execution Mode**: **Deterministic Local Simulation.** The remote Nebius endpoint was **not** queried (`gate_g0.connected = false`, `speedup_factor = null`). No remote execution, remote latency, or throughput is claimed here.
> **Dataset**: 50 training scenarios (300 verification episodes across 6 candidates; 410 episodes executed in total including held-out) + 30 held-out scenarios, seeded, drawn from the 9-category fault mix. Small samples: read the intervals.
> **Winning Patch Selected**: `patch_retry_policy`. Note that 4 candidates passed all guards with identical duplicate-action reduction on the training set; the winner was picked by the simplicity tie-breaker (shorter type string), not by superior measured performance.

### Baseline vs Hardened (Paired Held-Out Evaluation, same 30 scenarios)
Values are rate (count/denominator) [Wilson 95% CI].

| Metric (denominator) | Baseline (naive agent) | Hardened (`patch_retry_policy`) |
|---|---|---|
| **Duplicate-action rate** (all runs) | 10.0% (3/30) [3.5% - 25.6%] | 0.0% (0/30) [0.0% - 11.3%] |
| **Silent failure rate** (all runs) | 16.7% (5/30) [7.3% - 33.6%] | 0.0% (0/30) [0.0% - 11.3%] |
| **Claimed-success share among incorrect runs** (`conditional_false_success_rate`; incorrect runs only) | 87.5% (14/16) [64.0% - 96.5%] | 100.0% (16/16) [80.6% - 100.0%] |
| **Correct completion rate** (all runs) | 46.7% (14/30) [30.2% - 63.9%] | 46.7% (14/30) [30.2% - 63.9%] |
| **Legitimate completion rate** (legitimate-repeat + no-fault runs) | 77.8% (7/9) [45.3% - 93.7%] | 77.8% (7/9) [45.3% - 93.7%] |
| **Mean execution steps** | 2.93 | 2.90 |

With n=30 the upper confidence bound for a 0/30 rate is still 11.3%, so "0 observed" is not "proven zero".

### What the metrics mean (`runner.py` / `detectors.py`)
- **`silent_failure_rate`** (`counts.silent_fail` / `counts.total`): the final message claims a refund was issued **and** ground-truth state contradicts it: no refund exists, the claimed amount exceeds the refunded total, or a duplicate refund occurred.
- **`conditional_false_success_rate`** (claimed-success runs / `counts.cond_failed_total`): among runs that are **not correct**, the share whose final message claims a refund was issued. Despite its name it does **not** count false claims, and its denominator is incorrect runs, not all runs. It can be 100% while `silent_failure_rate` is 0%.
- **Correct completion**: no duplicate, no silent failure, no missed action, and no incorrect completion (`counts.correct` / `counts.total`).

### Why hardened correct completion did not improve (16/30 still incorrect)
The patch removes the fault-driven failures it targets (duplicates 3 → 0, silent failures 5 → 0, missed actions 2 → 0), but all 16 remaining hardened runs are *incorrect completions* with the expected refund **count** and a refund **total** that differs from the scenario's expected total. They claim a refund, which is why the conditional metric is 16/16. The cause is a benchmark mismatch, not a fault: the reference naive agent refunds half of orders totalling $50 or more, while single-refund scenarios expect the full order price (this also fails 2 of 2 no-fault controls). Until the agent or the scenario expectations are reconciled, correct-completion is capped by this and is not evidence about the patch.

### Implemented Verifier Guards (8 verifier guards)
The verifier deterministically evaluates candidates using 8 verifier guards:
1. **Incomplete/Missing Metrics Guard**: Rejects malformed or incomplete candidate metrics before evaluation.
2. **Regression Guard**: Protects against drops in baseline task completion rate beyond tolerance.
3. **Legitimate-Completion Guard**: Protects valid multi-item/partial refunds and no-fault actions against refusal.
4. **Duplicate-Action Guard**: Enforces non-increasing duplicate refund rates.
5. **False-Success Claim Guard**: Enforces non-increasing silent failure claims.
6. **Missed-Action Guard**: Caps omission rate increases.
7. **Over-Suppression Guard**: Rejects candidates that block legitimate operations.
8. **Cost Guard**: Caps mean execution steps within the configured factor of baseline.

### Verifier Guard Rejections Recorded
- `control_refuse_all`: **REJECTED** by Regression Guard (dropped completion) and Legitimate-Completion Guard.
- `control_blind_dedupe`: **REJECTED** by Legitimate-Completion Guard and Over-Suppression Guard (suppressed legitimate repeated partial refunds).

To reproduce this benchmark run:
```bash
py -c "from src.silentfail.orchestrator import Orchestrator; Orchestrator().run_full_pipeline(50, 30)"
py -m pytest -q
```
Counts, intervals, the winner, and guard outcomes are deterministic; timing fields and sample refund IDs vary per run.

### Known Limitations
- It is a local simulation only, so the committed results say nothing about Nebius/Nemotron behavior.
- `wrapper.read_back_verify`, `wrapper.retry_policy` and `combo.idempotency_and_read_back` currently share the same read-back-by-idempotency-key implementation, so they behave identically in this benchmark.
- Faults scheduled on a replayed idempotency key are consumed but never applied at the tool layer.
- The correct-completion cap described above.

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
