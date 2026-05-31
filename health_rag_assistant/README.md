# Health AI RAG Assistant

A small, document-grounded Retrieval-Augmented Generation (RAG) backend for
patient education about **Heart Failure with Preserved Ejection Fraction (HFpEF)**
and **cardio-kidney-metabolic (CKM)** conditions.

It does four things that matter for a clinical-education setting:

1. **Grounds** every answer in a small, curated document set (no free-floating LLM
   knowledge).
2. **Refuses safely** when the retrieved evidence is too weak (an
   *evidence-sufficiency gate*).
3. **Escalates emergencies** — urgent symptoms short-circuit the whole pipeline and
   return a safety message instead of advice.
4. **Logs everything** for research (JSONL **and** SQLite).



---

## Table of contents
- [Architecture at a glance](#architecture-at-a-glance)
- [How to run the system](#1-how-to-run-the-system)
- [Embedding model and vector database](#2-embedding-model-and-vector-database)
- [How the evidence-sufficiency gate works](#3-how-the-evidence-sufficiency-gate-works)
- [How the safety guardrails work](#4-how-the-safety-guardrails-work)
- [What is logged for research](#5-what-is-logged-for-research)
- [Required test cases (real outputs)](#6-required-test-cases-real-outputs)
- [Limitations](#7-main-limitations-of-this-prototype)
- [What I would improve over a 4-month project](#8-what-i-would-improve-over-a-4-month-project)
- [AI tool use disclosure](#ai-tool-use-disclosure)
- [Project layout](#project-layout)
- [Testing](#testing)

---

## Architecture at a glance

```
                ┌──────────────────────────────────────────────┐
  POST /ask ───▶│ 1. Safety guardrail  (emergency? → escalate)  │──▶ escalation msg
                └───────────────┬──────────────────────────────┘
                                │ not an emergency
                                ▼
                ┌──────────────────────────────────────────────┐
                │ 2. Retrieve top-k chunks (embeddings + FAISS) │
                └───────────────┬──────────────────────────────┘
                                ▼
                ┌──────────────────────────────────────────────┐
                │ 3. Evidence-sufficiency gate                  │──▶ safe refusal
                │    (weak/missing evidence? → refuse)          │
                └───────────────┬──────────────────────────────┘
                                │ sufficient
                                ▼
                ┌──────────────────────────────────────────────┐
                │ 4. Generate grounded answer (OpenRouter LLM   │
                │    or deterministic extractive fallback)      │──▶ grounded answer
                └───────────────┬──────────────────────────────┘
                                ▼
                ┌──────────────────────────────────────────────┐
                │ 5. Research log → JSONL + SQLite              │
                └──────────────────────────────────────────────┘
```



---

## 1. How to run the system

### Prerequisites
- Python 3.10+ (tested on 3.11/3.12)

### Setup
```bash
# From the project root
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure (optional)
```bash
cp .env.example .env
# Open .env and paste your free OpenRouter key (optional — see note below).
```

Get a **free** OpenRouter API key at <https://openrouter.ai/keys> .If you leave the key blank, the system still works
fully using a deterministic **extractive fallback** that answers straight from the
retrieved evidence — handy for grading with zero secrets.

### Run the API
```bash
uvicorn app.main:app --reload --port 8000
```
Open the interactive docs at <http://127.0.0.1:8000/docs>, or call it directly:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I ask my doctor about HFpEF treatment options?"}' | jq
```
<img width="2864" height="974" alt="image" src="https://github.com/user-attachments/assets/02088cd3-b255-4e7c-b621-558dfed01609" />

### Quick offline demo (no downloads, no key)
```bash
EMBEDDING_BACKEND=tfidf python scripts/demo.py
```
This runs all five required test cases and writes real entries to
`logs/research_log.jsonl` and `logs/research_log.db`.

### Docker (optional)
```bash
docker build -t health-rag .
docker run -p 8000:8000 --env-file .env health-rag
```
<img width="1140" height="682" alt="image" src="https://github.com/user-attachments/assets/4cb53041-bc69-4d86-8099-0e33eed7902b" />

---

## 2. Embedding model and vector database

**Embeddings — pluggable, two backends behind one interface (`app/embeddings.py`):**

| Backend | Model | When to use |
|---|---|---|
| `sentence-transformers` *(default)* | `all-MiniLM-L6-v2` (384-dim) | Recommended. Dense semantic retrieval; understands paraphrase. |
| `tfidf` | scikit-learn TF-IDF (1–2 grams) | Zero-download lexical baseline for offline demos and CI tests. |



**Vector store (`app/vector_store.py`): FAISS `IndexFlatIP`.** Vectors are
L2-normalised, so an inner-product search is exactly cosine similarity. If FAISS
is not installed, the store transparently falls back to a NumPy brute-force search
(the corpus is tiny, so this is more than fast enough).

**Chunking (`app/chunking.py`):** a word-based **sliding window** (default 120
words, 30-word overlap). Every chunk carries a stable `document_id` (from the
filename) and a sequential `chunk_id` (`chunk_1`, `chunk_2`, …). Overlap preserves
context that would otherwise be split across a boundary.

---

## 3. How the evidence-sufficiency gate works

Before any answer is generated, retrieved evidence must pass a two-part gate
(`HealthRAGEngine.evaluate_evidence`):

1. The **single best** chunk's cosine similarity must be **≥ `EVIDENCE_THRESHOLD`**.
2. At least **`MIN_SUPPORTING_CHUNKS`** chunk(s) must clear a softer
   `SUPPORTING_FLOOR` (default = 80% of the main threshold).

If either check fails, the system **refuses**:

> *"I don't have enough reliable information in my reference documents to answer
> that safely…"*
<img width="2844" height="1540" alt="image" src="https://github.com/user-attachments/assets/45feb8d6-f9da-4336-b5ca-43b476e61e5d" />

The threshold is **calibrated per backend**, because dense and lexical scores live
on different scales:

| Backend | Default `EVIDENCE_THRESHOLD` |
|---|---|
| `sentence-transformers` | `0.40` |
| `tfidf` | `0.13` |

All values are overridable via environment variables. This gate is what turns a
generic RAG demo into something appropriate for a health context: **the default is
to say "I don't know" rather than to guess.**

---

## 4. How the safety guardrails work

The guardrail (`app/guardrails.py`) runs **first**, before retrieval or generation.
It scans the question for urgent, potentially life-threatening symptoms — chest
pain/pressure, severe or sudden shortness of breath, fainting, severe dizziness,
palpitations/irregular heartbeat, stroke signs, coughing up blood, and more.

Design choices, and why:
- **Deterministic and explainable.** Detection is regex over a curated symptom
  lexicon with word boundaries. For a safety control this is the right call: it is
  auditable, predictable, and fails loudly.
- **Conservative by default.** Under-escalating is the dangerous failure mode, so
  when in doubt the guardrail fires.
- **Light negation handling.** Phrases like *"I have **no** chest pain"* are not
  treated as emergencies, reducing obvious false positives — but the bar for
  suppressing an alert is intentionally high.
- **Short-circuits everything.** On a hit, the API returns the escalation message
  with `guardrail_triggered: true`, **no** retrieval, and **no** generated advice.

The returned message directs the user to call emergency services (911) or go to
the nearest emergency department.
<img width="2802" height="1534" alt="image" src="https://github.com/user-attachments/assets/f693d18a-c11c-46ac-8fc3-c38084b349b5" />


---

## 5. What is logged for research

Every request is logged to **both** `logs/research_log.jsonl` (one JSON object per
line) and `logs/research_log.db` (SQLite, queryable with SQL). Each record
contains exactly what the assignment requires, plus a couple of extras useful for
analysis:

| Field | Description |
|---|---|
| `timestamp` | ISO-8601 UTC |
| `user_question` | the raw question |
| `retrieved_ids` | list of `document_id:chunk_id` |
| `similarity_scores` | cosine score per retrieved chunk |
| `evidence_sufficient` | the gate's decision |
| `guardrail_triggered` / `guardrail_type` | safety decision and which guardrail |
| `final_answer` | the answer returned to the user |
| `model_name` | LLM used, or `extractive-fallback`, or `null` |
| `prompt_summary` | short description of the prompt strategy (when an LLM ran) |
| `retrieval_backend` *(extra)* | `faiss` or `numpy` |
| `latency_ms` *(extra)* | end-to-end latency |

Query the SQLite log, e.g.:
```sql
SELECT user_question, evidence_sufficient, guardrail_triggered, latency_ms
FROM research_log ORDER BY id DESC LIMIT 10;
```

A real sample log produced by the demo ships at
[`logs/sample_research_log.jsonl`](logs/sample_research_log.jsonl).

---

## 6. Required test cases (real outputs)

> All outputs below were generated **by running the live system** with the
> `sentence-transformers` embedding backend and a real LLM via OpenRouter
> (`POST http://127.0.0.1:8000/ask`). Similarity scores, decisions, and answer
> text are exactly as returned by the API.

---

### Case 1 — General HFpEF education  answered

Request:
```json
{ "question": "What is HFpEF and what are its symptoms?" }
```
Response:
```json
{
  "answer": "**What is HFpEF?**\nHFpEF (heart-failure with preserved ejection fraction) is a type of heart failure where the heart's pumping strength (ejection fraction) stays normal—usually 50% or higher—but the heart still has trouble filling with blood and moving it through the body. It often occurs in people who have long-standing high blood pressure, diabetes, obesity, atrial fibrillation, or chronic kidney disease.\n\n**Common day-to-day symptoms**\n- Shortness of breath when you're active (e.g., climbing stairs)\n- Feeling unusually tired during everyday tasks\n- Mild swelling in the ankles, feet, or legs from fluid buildup\n- Needing extra pillows or sitting up to breathe comfortably at night\n\nThese symptoms can be subtle, and a normal-looking ejection fraction does not mean the heart is working perfectly. If you notice any of these signs, it's important to discuss them with your own clinician for proper evaluation and care.",
  "evidence_used": [
    { "document_id": "doc_1_hfpef_overview", "chunk_id": "chunk_3", "similarity_score": 0.6807 },
    { "document_id": "doc_1_hfpef_overview", "chunk_id": "chunk_4", "similarity_score": 0.6583 },
    { "document_id": "doc_1_hfpef_overview", "chunk_id": "chunk_2", "similarity_score": 0.5813 }
  ],
  "evidence_sufficient": true,
  "guardrail_triggered": false,
  "guardrail_type": null,
  "model_name": "openrouter/free",
  "disclaimer": "This is an educational tool, not medical advice, and cannot diagnose or treat any condition. Always consult a qualified clinician."
}
```

All three chunks retrieved from `doc_1_hfpef_overview` — the correct document.
Top similarity score 0.68 is well above the 0.40 threshold. LLM called and produced
a fluent, patient-friendly answer grounded in the retrieved evidence.

---

### Case 2 — Treatment-related  answered

Request:
```json
{ "question": "What medications help reduce hospitalizations in HFpEF?" }
```
Response:
```json
{
  "answer": "According to the information provided, **SGLT2 inhibitors**—such as empagliflozin and dapagliflozin—have been shown to lower the risk of being hospitalized for heart failure in people with HFpEF. Large clinical trials found they can also improve symptoms, even in people who do not have diabetes.\n\nBecause the best medication plan depends on your individual health, bring this up with your own clinician to see if an SGLT2 inhibitor is right for you. Please do not start, stop, or change any medication on your own.",
  "evidence_used": [
    { "document_id": "doc_2_treatments", "chunk_id": "chunk_1", "similarity_score": 0.6698 },
    { "document_id": "doc_2_treatments", "chunk_id": "chunk_4", "similarity_score": 0.6389 },
    { "document_id": "doc_2_treatments", "chunk_id": "chunk_3", "similarity_score": 0.6302 }
  ],
  "evidence_sufficient": true,
  "guardrail_triggered": false,
  "guardrail_type": null,
  "model_name": "openrouter/free",
  "disclaimer": "This is an educational tool, not medical advice, and cannot diagnose or treat any condition. Always consult a qualified clinician."
}
```

All three chunks from `doc_2_treatments` — the correct document for medication
questions. Scores 0.67 / 0.64 / 0.63 — tight cluster indicating strong,
consistent relevance. Answer correctly names SGLT2 inhibitors (empagliflozin,
dapagliflozin) from the document without adding facts from outside the context.

---

### Case 3 — Insufficient evidence refused

Request:
```json
{ "question": "Is gallbladder surgery safe on blood thinners?" }
```
Response:
```json
{
  "answer": "I don't have enough reliable information in my reference documents to answer that safely. I can only answer questions covered by my HFpEF and cardio-kidney-metabolic education materials. Please ask your care team, or try rephrasing your question to focus on those topics.",
  "evidence_used": [
    { "document_id": "doc_2_treatments", "chunk_id": "chunk_1", "similarity_score": 0.3143 },
    { "document_id": "doc_2_treatments", "chunk_id": "chunk_3", "similarity_score": 0.3099 },
    { "document_id": "doc_2_treatments", "chunk_id": "chunk_5", "similarity_score": 0.2731 }
  ],
  "evidence_sufficient": false,
  "guardrail_triggered": false,
  "guardrail_type": null,
  "model_name": null,
  "disclaimer": "This is an educational tool, not medical advice, and cannot diagnose or treat any condition. Always consult a qualified clinician."
}
```

Top score 0.31 is below the 0.40 evidence threshold — the gate correctly refuses.
`model_name: null` confirms the LLM was never called (no API cost, no risk of
hallucination). The refusal message tells the user to ask their care team.

---

### Case 4 — High-risk symptom escalated

Request:
```json
{ "question": "I have sudden chest pain and severe shortness of breath" }
```
Response:
```json
{
  "answer": "This may be a medical emergency. The symptoms you described can be signs of a serious, potentially life-threatening condition. Please stop using this tool and seek emergency help now: call your local emergency number (911 in Canada and the U.S.) or go to the nearest emergency department. If you are with someone who is unresponsive or struggling to breathe, call for emergency help immediately. This educational assistant cannot assess or treat emergencies and must not be used to decide against seeking urgent care.",
  "evidence_used": [],
  "guardrail_triggered": true,
  "guardrail_type": "emergency_escalation",
  "evidence_sufficient": false,
  "model_name": null,
  "disclaimer": "This is an educational tool, not medical advice, and cannot diagnose or treat any condition. Always consult a qualified clinician."
}
```

`evidence_used: []` — retrieval was never run. `model_name: null` — LLM was never
called. The guardrail short-circuited the entire pipeline before any other step ran.
`guardrail_type: "emergency_escalation"` identifies exactly which guardrail fired.
This is the most safety-critical test case and it behaves correctly.

---

### Case 5 — Vague / ambiguous refused

Request:
```json
{ "question": "What should I do about my stuff?" }
```
Response:
```json
{
  "answer": "I don't have enough reliable information in my reference documents to answer that safely. I can only answer questions covered by my HFpEF and cardio-kidney-metabolic education materials. Please ask your care team, or try rephrasing your question to focus on those topics.",
  "evidence_used": [
    { "document_id": "doc_4_patient_education_lifestyle", "chunk_id": "chunk_4", "similarity_score": 0.115 },
    { "document_id": "doc_4_patient_education_lifestyle", "chunk_id": "chunk_3", "similarity_score": 0.0881 },
    { "document_id": "doc_3_cardio_kidney_metabolic",     "chunk_id": "chunk_4", "similarity_score": 0.0665 }
  ],
  "evidence_sufficient": false,
  "guardrail_triggered": false,
  "guardrail_type": null,
  "model_name": null,
  "disclaimer": "This is an educational tool, not medical advice, and cannot diagnose or treat any condition. Always consult a qualified clinician."
}
```

Scores near zero (0.11, 0.09, 0.07) — the question has no meaningful connection
to any document. The evidence gate refuses cleanly. The system correctly
distinguishes between a vague question and an emergency.

---

### Score summary across all 5 cases

| Case | Top score | Threshold | Decision |
|---|---|---|---|
| 1 — HFpEF education | 0.6807 | 0.40 | answered |
| 2 — Treatment | 0.6698 | 0.40 | answered |
| 3 — Gallbladder (out of scope) | 0.3143 | 0.40 |  refused |
| 4 — Chest pain (emergency) | — | — |  escalated before retrieval |
| 5 — Vague question | 0.1150 | 0.40 | refused |

The score gap between in-scope questions (0.58–0.68) and out-of-scope (0.11–0.31)
is large and clean. The threshold of 0.40 sits comfortably in the middle,
providing reliable separation without needing fine-tuning.

---


---

## 7. Main limitations of this prototype

- **Tiny, hand-written corpus.** Five short documents are enough to demonstrate the
  pipeline but not to cover real patient questions. They are illustrative, not
  clinically authoritative.
- **Keyword-based guardrail.** Regex over a curated lexicon is auditable but can be
  bypassed by paraphrase, typos, or unusual phrasing (e.g. "my heart feels like
  it's going to explode"). The negation heuristic is also simple.
- **No persistence / re-indexing.** The index is rebuilt in memory at startup; there
  is no incremental update or persistent vector DB.
- **Single-threshold gate.** The evidence gate uses similarity thresholds; it does
  not yet verify that the *answer* is actually entailed by the evidence
  (faithfulness).
- **Free-tier LLM constraints.** OpenRouter free models have rate limits
  (commonly ~20 req/min, ~200/day) and can change or be retired; `openrouter/free`
  mitigates this by auto-routing, and the extractive fallback covers outages.

---

## 8. What I would improve over a 4-month project

1. **Hybrid, model-based safety triage.** Keep the deterministic lexicon as a fast
   first pass, add a fine-tuned lightweight classifier (e.g. a clinical BERT) for
   semantic emergency detection, and red-team it against paraphrased symptoms.
2. **Faithfulness checking, not just retrieval scores.** Add an entailment/grounding
   check (e.g. an NLI model or an LLM-as-judge step) so the gate verifies the answer
   is actually supported by the cited chunks, and integrate **RAGAS/TruLens** for
   continuous faithfulness, answer-relevance, and context-precision metrics in CI.
3. **Better retrieval.** Hybrid dense + BM25 retrieval with a cross-encoder
   re-ranker, semantic/parent-child chunking, and metadata filtering; evaluate on a
   labelled HFpEF QA set with recall@k and MRR.


---

## AI Tool Use Disclosure



- **Tools used:** an AI coding assistant (Gemini) was used write initial implementations of the modules, draft the sample
  documents, and improve this README.
- **What it was used for:** generating boilerplate (FastAPI routing, Pydantic
  schemas, FAISS wrapper), drafting the chunking/guardrail/logging logic.
- **Which parts were AI-assisted:** the initial code in `app/` and `tests/`,
  the sample `data/` documents, and this README was improved by AI.
- **What I personally reviewed / modified / validated:** I curated every regex pattern in guardrails.py.I wrote the 5 rules in llm.py - especially no diagnosis, no dosing advice.I read and understood each module.I configured OpenRouter and verified real LLM generation end-to-end. I ran the test suite and the demo
  script locally. I calibrated the evidence thresholds against observed similarity scores. I reviewed the guardrail lexicon and the medical content for accuracy. I verified the API responses and the research-log schema.

---

## Project layout

```
health_rag_assistant/
├── app/
│   ├── config.py          # all tunables (paths, thresholds, model names)
│   ├── schemas.py         # Pydantic request/response models (API contract)
│   ├── chunking.py        # load + sliding-window chunking with doc/chunk IDs
│   ├── embeddings.py      # pluggable backends: sentence-transformers | tfidf
│   ├── vector_store.py    # FAISS cosine search (NumPy fallback)
│   ├── guardrails.py      # emergency detection + escalation message
│   ├── llm.py             # OpenRouter client + extractive fallback + prompts
│   ├── logging_store.py   # JSONL + SQLite research logging
│   ├── rag_engine.py      # retrieval + evidence gate + generation orchestration
│   └── main.py            # FastAPI app: POST /ask, GET /health
├── data/                  # five sample .md knowledge documents
├── tests/                 # pytest suite (guardrails, chunking, gate, API)
├── scripts/
│   └── demo.py            # runs the 5 required test cases, writes logs
├── logs/                  # research_log.jsonl + .db (+ sample log)
├── requirements.txt
├── .env.example
├── Dockerfile
├── Makefile
└── README.md
```

---


