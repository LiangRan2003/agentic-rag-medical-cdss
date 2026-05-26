import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 采用 BAAI 的 BGE 模型，中文检索效果非常好，也是简历上的一个亮点（本地部署 Embedding）
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CHROMA_PERSIST_DIR = "./chroma_db"

def get_embeddings():
    """获取文本向量化模型"""
    print(f"正在加载 Embedding 模型: {EMBEDDING_MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # model_kwargs={'device': 'cpu'}, # 如果有GPU可以去掉或改成cuda
        encode_kwargs={'normalize_embeddings': True} # BGE 模型建议 normalize
    )
    return embeddings

def init_vector_store(data_dir="./data"):
    """
    初始化并构建向量数据库。
    如果 persist_dir 已存在并且有数据，可以直接读取。
    """
    embeddings = get_embeddings()
    
    if os.path.exists(CHROMA_PERSIST_DIR) and len(os.listdir(CHROMA_PERSIST_DIR)) > 0:
        print("发现已存在的向量数据库，正在加载...")
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        return vectorstore
        
    print("未发现现有向量数据库，开始从 PDF 构建...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"请将 PDF 文件放入 {data_dir} 文件夹中。")
        return None

    files = [f for f in os.listdir(data_dir) if f.endswith('.pdf') or f.endswith('.docx')]
    if not files:
        print(f"{data_dir} 文件夹中没有找到 PDF 或 DOCX 文件！")
        return None

    docs = []
    for file in files:
        print(f"正在处理: {file}")
        file_path = os.path.join(data_dir, file)
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        docs.extend(loader.load())

    # 文本分块：由于中文排版特点，使用 RecursiveCharacterTextSplitter 比较合适
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    print(f"文档切分完成，共 {len(splits)} 个 chunk，正在存入向量数据库...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("向量数据库构建完成！")
    return vectorstore

if __name__ == "__main__":
    init_vector_store()
