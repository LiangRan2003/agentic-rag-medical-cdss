import streamlit as st
import os
from dotenv import load_dotenv
from src.vector_store import init_vector_store
from src.workflow import build_workflow

# 加载环境变量
load_dotenv()

st.set_page_config(page_title="垂直领域 Agentic RAG 助手", layout="wide")

st.title("⚕️ 感染科辅助诊疗智能体 (CDSS Agent)")
st.markdown("这是一个包含 **检索 (Retriever)**、**生成 (Synthesizer)** 和 **幻觉校验 (Validator)** 多 Agent 协同的医疗 RAG 系统。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 配置区")
    api_key = st.text_input("输入 DeepSeek API Key", type="password", value=os.environ.get("DEEPSEEK_API_KEY", ""))
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        
    st.markdown("---")
    st.subheader("📚 知识库管理")
    st.info("请将医学指南 PDF/DOCX 放在项目 `data` 文件夹下。")
    if st.button("构建/加载向量数据库"):
        with st.spinner("正在处理文档并构建向量数据库... (第一次运行需要提取并计算文本向量，大概需要1-3分钟，请耐心等待)"):
            vs = init_vector_store()
            if vs:
                st.session_state['retriever'] = vs.as_retriever(search_kwargs={"k": 3})
                st.success("向量数据库加载成功！")
            else:
                st.error("加载失败，请检查 data 文件夹是否有文件。")

# --- 主交互区 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入您要咨询的医学问题（例如：中国社区获得性肺炎的常见致病菌有哪些？）"):
    if 'retriever' not in st.session_state:
        st.warning("请先在侧边栏加载向量数据库！")
    elif not os.environ.get("DEEPSEEK_API_KEY"):
        st.warning("请先配置 DeepSeek API Key！")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = st.container()
            workflow = build_workflow(st.session_state['retriever'])
            
            inputs = {"question": prompt}
            
            # 使用 spinner 显示进度
            with st.spinner("Agent 思考中..."):
                final_generation = ""
                # stream 获取每个节点的执行状态
                for output in workflow.stream(inputs):
                    for key, value in output.items():
                        if key == "retrieve":
                            status_container.info("🔍 Retriever Agent: 正在从向量库中检索相关文档...")
                        elif key == "grade_documents":
                            status_container.info("⚖️ Validator Agent: 正在评估检索文档的相关性...")
                        elif key == "generate":
                            status_container.info("✍️ Synthesizer Agent: 正在生成回答，并进行幻觉校验...")
                            if "generation" in value:
                                final_generation = value["generation"]
            
            st.markdown(final_generation)
            st.session_state.messages.append({"role": "assistant", "content": final_generation})
