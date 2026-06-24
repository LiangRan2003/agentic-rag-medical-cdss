# Test Report

## Summary

Added a deterministic pytest suite for the Agentic RAG workflow. The tests use lightweight mocks for LangChain and LangGraph so the workflow logic can be validated without downloading models, calling DeepSeek, or building the Chroma vector store.

## What Is Covered

- Retriever node passes the user question to the retriever and returns the retrieved documents.
- Document grading filters irrelevant documents and marks when further search may be needed.
- Hallucination-check routing returns:
  - `useful` for supported answers.
  - `not supported` for unsupported answers.
- Workflow graph construction registers the expected nodes, edges, and conditional routing.

## Why This Matters

The project is an agentic RAG system where correctness depends on routing between retrieval, grading, generation, and validation. These tests exercise the control-flow logic directly while avoiding external model calls, making them fast enough for local development and CI.

## Verification

Command:

```powershell
python -m pytest -q
```

Result:

```text
4 passed
```

## Files Added

- `tests/conftest.py`
- `tests/test_workflow_nodes.py`
