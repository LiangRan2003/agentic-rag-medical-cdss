import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import workflow


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def invoke(self, question):
        self.queries.append(question)
        return self.documents


def test_retrieve_uses_question_and_returns_documents():
    docs = [Document("antibiotic guideline")]
    retriever = FakeRetriever(docs)

    result = workflow.retrieve({"question": "CAP treatment"}, retriever)

    assert retriever.queries == ["CAP treatment"]
    assert result == {"documents": docs, "question": "CAP treatment"}


def test_grade_documents_filters_irrelevant_context(monkeypatch):
    monkeypatch.setattr(workflow, "get_llm", lambda: object())
    workflow.PromptTemplate.responder = staticmethod(
        lambda inputs: "yes" if "pneumonia" in inputs["context"] else "no"
    )
    docs = [Document("pneumonia antibiotics"), Document("unrelated billing text")]

    result = workflow.grade_documents({"question": "How to treat pneumonia?", "documents": docs})

    assert result["documents"] == [docs[0]]
    assert result["web_search"] == "Yes"


def test_check_hallucination_routes_supported_and_unsupported(monkeypatch):
    monkeypatch.setattr(workflow, "get_llm", lambda: object())
    state = {
        "question": "What is supported?",
        "documents": [Document("fact: use guideline evidence")],
        "generation": "use guideline evidence",
    }

    workflow.PromptTemplate.responder = staticmethod(lambda inputs: "yes")
    assert workflow.check_hallucination(state) == "useful"

    workflow.PromptTemplate.responder = staticmethod(lambda inputs: "no")
    assert workflow.check_hallucination(state) == "not supported"


def test_build_workflow_registers_expected_nodes_and_edges():
    app = workflow.build_workflow(FakeRetriever([]))

    assert {"retrieve", "grade_documents", "generate"} <= set(app.nodes)
    assert ("__start__", "retrieve") in app.edges
    assert ("retrieve", "grade_documents") in app.edges
    assert ("grade_documents", "generate") in app.edges
    assert app.conditional_edges[0][2] == {
        "useful": "__end__",
        "not supported": "generate",
    }
