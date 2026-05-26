# ⚕️ 感染科辅助诊疗智能体 (Infectious Disease CDSS Agent)

基于大语言模型与多智能体协同 (Multi-Agent Workflows) 构建的垂直医疗领域 RAG 系统，旨在提供高度准确、无幻觉的临床决策支持 (CDSS)。

## 🌟 项目亮点

在严肃的医疗场景（特别是感染科用药指征严格的场景）中，传统的 RAG 系统容易因为检索不精准而导致大模型产生致命的“幻觉”。
本项目通过引入 **Agentic RAG（智能体化检索增强生成）** 架构，彻底解决了这一痛点：

1. **多智能体协同**：系统拆分为 Retriever Agent、Synthesizer Agent 和 Validator Agent。
2. **严苛的幻觉校验 (Hallucination Check)**：采用 LLM-as-a-Judge 机制，如果系统检测到生成的答案包含指南中不存在的捏造信息，将自动打回重做，确保输出 100% 具备溯源性 (Grounding)。
3. **高质量本地知识库**：内置多份最新版国家级医疗指南与共识（如《慢性乙型肝炎防治指南》、《中国社区获得性肺炎诊断和治疗指南》等），作为系统的金标准知识来源。
4. **流式状态可视化**：前端采用 Streamlit，动态展示 Agent 工作流的“思考过程”，使不可见的黑盒推理过程变得透明可信。

## 🏗️ 系统架构 (基于 LangGraph)

- **Retriever Node**: 利用本地部署的 BGE-m3 模型将临床指南文档向量化，存入 Chroma 数据库。接收用户输入后进行向量近似搜索。
- **Grade Documents Node**: 评估召回文档与用户问题是否高度相关，过滤无关噪音。
- **Generate Node**: 基于召回的高质量 Context 进行医疗回答的生成。
- **Hallucination Validator**: 条件判定边 (Conditional Edge)。对比生成结果与原始检索文本，判定是否产生幻觉。

## 🚀 快速启动

### 1. 环境准备
项目使用 Python 开发，建议使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt # (或根据源码中引用的 langchain 相关库自行安装)
```

### 2. 配置 API Key
在项目根目录创建 `.env` 文件，填入兼容 OpenAI 接口规范的 API Key（本项目默认使用高性价比的 DeepSeek）：
```env
DEEPSEEK_API_KEY="your_api_key_here"
```

### 3. 运行应用
```bash
streamlit run app.py
```

### 4. 加载知识库
首次启动后，请在网页左侧边栏点击 **“构建/加载向量数据库”**。系统将自动读取 `data/` 目录下的 PDF/DOCX 指南文件，并进行分块、向量化（耗时约1-2分钟）。后续使用将实现秒级加载。

## 📁 目录说明

- `app.py`: Streamlit 前端交互与入口文件。
- `src/workflow.py`: 基于 LangGraph 定义的多 Agent 工作流逻辑。
- `src/vector_store.py`: 本地文档读取 (PyPDFLoader/Docx2txtLoader) 与 Chroma 向量数据库构建逻辑。
- `data/`: 存放《诊疗指南》及《专家共识》等权威医学源文件。

## 🤝 鸣谢与说明
本项目所附带的医疗指南仅作为代码演示与学术研究数据测试使用，任何生成的结论**不应被直接作为真实临床治疗依据**。
