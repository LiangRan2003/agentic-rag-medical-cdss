import os
from typing import List, Dict, Any, TypedDict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

# 定义 Graph 的 State
class GraphState(TypedDict):
    """
    表示图的状态。
    """
    question: str
    generation: str
    documents: List[Document]
    web_search: str # 'Yes' or 'No' 决定是否需要回退到网络搜索或重新检索

# --- 节点定义 ---

def get_llm():
    """获取大语言模型实例 (使用 DeepSeek 兼容 OpenAI 接口)"""
    # 需确保环境变量中存在 DEEPSEEK_API_KEY
    api_key = os.environ.get("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")
    return ChatOpenAI(
        model="deepseek-chat", 
        api_key=api_key, 
        base_url="https://api.deepseek.com",
        temperature=0
    )

def retrieve(state: GraphState, retriever):
    """
    检索文档节点
    """
    print("--- 正在检索文档 ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """
    基于检索文档生成回答的节点 (Synthesizer Agent)
    """
    print("--- 正在生成回答 ---")
    question = state["question"]
    documents = state["documents"]
    
    # 拼接文档内容
    context = "\n\n".join([doc.page_content for doc in documents])
    
    template = """你是专业的感染科辅助诊疗智能体 (CDSS Agent)。请使用以下检索到的医学指南和共识来回答用户的问题。
如果你不知道答案，或者参考材料中没有提及，请直接说明你不知道，切勿编造（产生幻觉）。
保持回答的专业性、客观性和逻辑性。

参考材料：
{context}

用户问题：
{question}

专业回答："""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = get_llm()
    rag_chain = prompt | llm | StrOutputParser()
    
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state: GraphState):
    """
    判断检索到的文档是否与问题相关的节点
    """
    print("--- 校验文档相关性 ---")
    question = state["question"]
    documents = state["documents"]
    
    llm = get_llm()
    
    system = """你是一个评分员，负责评估检索到的文档与用户问题的相关性。 \n 
    如果文档包含与用户问题相关的关键字或语义含义，请将其评为相关。 \n
    给出一个二元分数 'yes' 或 'no' 来表明文档是否与问题相关。请只输出 'yes' 或 'no'，不要输出任何其他内容。"""
    grade_prompt = PromptTemplate(template=system + "\n检索到的文档: \n\n {context} \n\n 用户问题: {question}", input_variables=["context", "question"])
    retrieval_grader = grade_prompt | llm | StrOutputParser()
    
    filtered_docs = []
    web_search = "No"
    for d in documents:
        grade = retrieval_grader.invoke({"question": question, "context": d.page_content}).strip().lower()
        if "yes" in grade:
            filtered_docs.append(d)
        else:
            web_search = "Yes" # 只要有一个不相关，就标记可能需要进一步检索（这里简化逻辑）
            
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def check_hallucination(state: GraphState) -> str:
    """
    校验生成的回答是否产生幻觉的条件边逻辑 (Validator Agent)
    返回 'useful' (没有幻觉且回答了问题), 'not supported' (产生幻觉), 或 'not useful' (没回答问题)
    """
    print("--- 正在检测幻觉 ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    llm = get_llm()
    system = """你是一名事实核查员，负责评估大型语言模型的回答是否完全基于检索到的事实。 \n
    如果回答完全基于提供的事实，没有捏造（幻觉），请输出 'yes'，否则输出 'no'。请只输出 'yes' 或 'no'，不要输出任何其他内容。"""
    hallucination_prompt = PromptTemplate(template=system + "\n事实文档: \n\n {documents} \n\n LLM的回答: {generation}", input_variables=["documents", "generation"])
    hallucination_grader = hallucination_prompt | llm | StrOutputParser()
    
    grade = hallucination_grader.invoke({"documents": [d.page_content for d in documents], "generation": generation}).strip().lower()
    
    if "yes" in grade:
        print("--- 判定：未产生幻觉 ---")
        return "useful"
    else:
        print("--- 判定：存在幻觉风险，需要重新生成 ---")
        return "not supported"

# --- 图的构建 ---
def build_workflow(retriever):
    workflow = StateGraph(GraphState)
    
    # 包装 retrieve 节点，使其能接收 retriever
    def retrieve_node(state):
        return retrieve(state, retriever)
        
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    
    # 定义边
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    
    # 定义条件边：检查幻觉，如果有幻觉要求重新生成（简化版，实际中可以加最大重试次数）
    workflow.add_conditional_edges(
        "generate",
        check_hallucination,
        {
            "useful": END,
            "not supported": "generate", # 如果有幻觉，让它根据原始文档重新生成
        }
    )
    
    app = workflow.compile()
    return app
