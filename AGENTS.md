# HallucinationRadar — AGENTS.md

## Setup

```bash
pip install langchain-ollama langchain-community wikipedia
```

Requires **Ollama** with models:
- `deepseek-r1:7b` — answer generation (`nodes/answer.py`)
- `llama3.1` — everything else (claims, retrieval, verify, report)

No test framework, no formatter, no type checker, no `requirements.txt`. Python 3.12.2.

## Pipeline

`mains.py` runs only `answer → claims`. Nodes for `retrieval → verify → score → report` exist but are **not connected**.

Pipeline state is a plain `dict` with keys: `query → answer → claims → evidence → verdicts → score → report`. No TypedDict/Pydantic model.

Entrypoint: `python mains.py` — currently queries "What is an LLM?".

## Node conventions

- All LLM prompts use `temperature=0`.
- `answer.py` imports from `langchain_ollama`; all other nodes use `langchain_community.chat_models`.
- `nodes/` has no `__init__.py` — works via implicit namespace packages (Python 3.3+).

### verify.py JSON contract

Expects LLM to return JSON with keys `verdict`, `confidence`, `reason`. Strips markdown fences if present.

Verdict values: `"supported" | "contradicted" | "unverifiable"` — shared across verify → score → report.

**Dead code bug**: verify.py line 16 checks `snippets == ["No evidence found."]` but retrieval.py actually stores `" nothing found apparently T_T . ggs mate Y_Y"` on Wikipedia failure. The fallback string goes through LLM verification instead of being short-circuited.

### score.py formula

`(base + confidence) / 2` per claim, where `base` is `{supported: 1.0, unverifiable: 0.5, contradicted: 0.0}`.

## Dataset

`datasetPart/` — management textbook chunks for local evidence.

`datasetPart/extractingDataFrmTb.py` chunks `.txt` into JSON with section metadata. **Windows path gotcha**: lines 91–94 contain hardcoded `C://Users//...` paths — update to relative paths before running on Linux.

`1.MANAGEMENT_chunks.json` already has 282 pre-computed chunks.

## Dead / experimental code

`testingLLM/` — mostly commented-out ReAct agent experiments (`testingLLM_olama.py`). The uncommented portion at the bottom (StructuredTool agent) is unrelated and disconnected from the pipeline. Ask before deleting.

`bluePrint_2.0.txt` — aspirational LangGraph design; the actual code never adopted it.
