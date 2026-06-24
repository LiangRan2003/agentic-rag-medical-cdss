import sys
import types


class Document:
    def __init__(self, page_content):
        self.page_content = page_content


class _FakePromptTemplate:
    responder = staticmethod(lambda inputs: "yes")

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def __or__(self, other):
        return _FakeChain(self.responder)


class _FakeChain:
    def __init__(self, responder):
        self.responder = responder

    def __or__(self, other):
        return self

    def invoke(self, inputs):
        return self.responder(inputs)


class _FakeStateGraph:
    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, source, target):
        self.edges.append((source, target))

    def add_conditional_edges(self, source, condition, mapping):
        self.conditional_edges.append((source, condition, mapping))

    def compile(self):
        return self


def pytest_configure():
    docs_module = types.ModuleType("langchain_core.documents")
    docs_module.Document = Document

    prompts_module = types.ModuleType("langchain_core.prompts")
    prompts_module.PromptTemplate = _FakePromptTemplate

    parsers_module = types.ModuleType("langchain_core.output_parsers")
    parsers_module.StrOutputParser = lambda: object()

    openai_module = types.ModuleType("langchain_openai")
    openai_module.ChatOpenAI = lambda *args, **kwargs: object()

    graph_module = types.ModuleType("langgraph.graph")
    graph_module.START = "__start__"
    graph_module.END = "__end__"
    graph_module.StateGraph = _FakeStateGraph

    sys.modules.setdefault("langchain_core.documents", docs_module)
    sys.modules.setdefault("langchain_core.prompts", prompts_module)
    sys.modules.setdefault("langchain_core.output_parsers", parsers_module)
    sys.modules.setdefault("langchain_openai", openai_module)
    sys.modules.setdefault("langgraph.graph", graph_module)
