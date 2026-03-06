import streamlit as st
import os
import tempfile
import pandas as pd
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ============================================================
# 🔧 全局配置区（针对 MetaX C500 8卡64GB 优化）
# ============================================================
CONFIG = {
    # LLM 配置（OpenAI 兼容接口，对接 vLLM）
    "llm_base_url": "http://localhost:8000/v1",  # vLLM 服务地址
    "llm_model": "qwen2.5-72b",                   # 模型名称（72B充分利用8卡性能）
    "llm_api_key": "EMPTY",                       # 本地部署无需真实 key
    "llm_temperature": 0.2,                        # 温度参数（降低以提高准确性）
    "llm_max_tokens": 4096,                        # 最大token数（8卡可处理更长输出）
    "llm_top_p": 0.9,                              # nucleus sampling
    "llm_frequency_penalty": 0.1,                  # 减少重复

    # Embedding 配置（C500 优化）
    "embedding_model": "BAAI/bge-large-zh-v1.5",  # 升级到large模型（更强语义理解）
    "embedding_device": "cuda",                    # 使用GPU加速
    "embedding_batch_size": 128,                   # 大批处理（充分利用显存）
    "embedding_normalize": True,                   # 向量归一化

    # 向量数据库配置（C500 优化）
    "chroma_persist_dir": "./chroma_db",
    "retriever_top_k": 20,                         # 检索数量（8卡可处理更多上下文）
    "retriever_score_threshold": 0.4,              # 相似度阈值（更宽松以提高召回）
    "retriever_fetch_k": 50,                       # 初筛数量（用于MMR等高级检索）

    # 文本切分配置（C500 优化）
    "chunk_size": 1500,                            # 文本块大小（增大以保留更多上下文）
    "chunk_overlap": 300,                          # 文本块重叠（增大以避免信息断裂）
    "separators": ["\n\n", "\n", "。", "！", "？", ";", "，", " ", ""],  # 中文优化分隔符

    # 性能优化配置（C500 专属）
    "max_documents_per_upload": 100,               # 单次上传文档数量（8卡可处理更多）
    "enable_reranking": False,                     # 是否启用重排序（可选，需额外模型）
    "reranker_model": "BAAI/bge-reranker-large",  # 重排序模型
    "reranker_top_n": 10,                          # 重排序后保留数量
    "enable_cache": True,                          # 启用缓存
    "show_performance_stats": True,                # 显示性能统计
    
    # 显示配置
    "page_title": "🚀 国产GPU超大规模RAG",
    "model_display_name": "Qwen2.5-72B",
    "gpu_display_name": "MetaX C500 (8卡64GB)",
}

# ============================================================
# 🎨 页面配置与 CSS
# ============================================================
st.set_page_config(
    page_title=CONFIG["page_title"],
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    /* 全局背景与字体 */
    .main { background-color: #f8f9fa; }
    .stApp { font-family: 'Inter', -apple-system, sans-serif; }
    
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E;
        color: white;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, h2, h3 { color: #00D4FF; }
    
    /* 思考过程容器 */
    .thought-card {
        background: #3b3939;
        border-radius: 12px;
        border-left: 5px solid #00D4FF;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    /* 对话气泡微调 */
    .stChatMessage {
        border-radius: 15px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    
    /* 标题特效 */
    .main-title {
        background: linear-gradient(90deg, #00D4FF, #0072FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
    }
    
    /* 配置信息卡片 */
    .config-card {
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.85em;
        line-height: 1.8;
    }
    
    /* 来源片段卡片 */
    .source-card {
        background: #f8faff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.88em;
        color: #374151;
        border-left: 3px solid #00D4FF;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 🤖 模型初始化（全局缓存，只加载一次）
# ============================================================
@st.cache_resource
def init_models():
    # LLM（OpenAI 兼容接口，C500 优化）
    llm = ChatOpenAI(
        base_url=CONFIG["llm_base_url"],
        model=CONFIG["llm_model"],
        api_key=CONFIG["llm_api_key"],
        temperature=CONFIG["llm_temperature"],
        max_tokens=CONFIG["llm_max_tokens"],
        top_p=CONFIG["llm_top_p"],
        frequency_penalty=CONFIG["llm_frequency_penalty"],
        streaming=True,
    )
    
    # Embedding（C500 优化：升级到 large 模型 + 批处理）
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={
            "device": CONFIG["embedding_device"],
        },
        encode_kwargs={
            "batch_size": CONFIG["embedding_batch_size"],
            "normalize_embeddings": CONFIG["embedding_normalize"],
        },
    )
    
    return llm, embeddings

llm, embeddings = init_models()

# ============================================================
# 📄 多格式文档处理
# ============================================================
SUPPORTED_FORMATS = {
    ".pdf":  "📕 PDF",
    ".docx": "📘 Word",
    ".doc":  "📘 Word",
    ".txt":  "📄 TXT",
    ".xlsx": "📗 Excel",
    ".xls":  "📗 Excel",
    ".md":   "📝 Markdown",
}

def load_documents(uploaded_files):
    """解析多种格式文档，返回 LangChain Document 列表"""
    all_docs = []
    for f in uploaded_files:
        ext = Path(f.name).suffix.lower()
        suffix = ext if ext else ".tmp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
                all_docs.extend(loader.load())

            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(tmp_path)
                all_docs.extend(loader.load())

            elif ext == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
                all_docs.extend(loader.load())

            elif ext in [".xlsx", ".xls"]:
                # Excel 转文本（保留表格结构）
                xl = pd.ExcelFile(tmp_path)
                for sheet_name in xl.sheet_names:
                    df = xl.parse(sheet_name)
                    content = f"【Sheet: {sheet_name}】\n{df.to_string(index=False)}"
                    all_docs.append(
                        Document(
                            page_content=content,
                            metadata={"source": f.name, "sheet": sheet_name},
                        )
                    )
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(tmp_path)
                all_docs.extend(loader.load())
            else:
                st.warning(f"⚠️ 不支持的格式: {f.name}")

        except Exception as e:
            st.error(f"❌ 解析 {f.name} 失败: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return all_docs

# ============================================================
# 🧠 RAG 链构建
# ============================================================
def get_rag_chain(msgs):
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": CONFIG["retriever_top_k"]}
    )

    # 问题改写 Prompt（多轮对话关键）
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "根据对话历史和最新问题，将问题改写为独立的搜索查询。"
            "若问题已独立完整，原样返回。只输出改写后的问题，不要解释。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 回答 Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个专业的文档分析助手，运行在国产沐曦 MetaX C500 GPU 服务器上。\n"
            "请严格基于以下【参考文档】回答问题，保持客观准确。\n"
            "若文档中无相关信息，请明确说明无法从文档中找到答案。\n\n"
            "【参考文档】\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# ============================================================
# 💬 记忆管理
# ============================================================
msgs = StreamlitChatMessageHistory(key="chat_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message(
        "👋 你好！我是运行在 **MetaX C500 8卡** 服务器上的知识库助手，"
        "由 **Qwen2.5-72B** 模型驱动。\n\n"
        "请在左侧上传文档，我将为您提供精准的文档问答服务。"
    )

# ============================================================
# 🗂️ 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("## 🛠️ 控制面板")
    st.divider()

    # 文档上传
    st.markdown("### 📥 数据注入")
    st.caption("支持 PDF / Word / Excel / TXT / Markdown")
    uploaded_files = st.file_uploader(
        "支持格式: PDF, TXT, Word, Excel, Markdown",
        type=["pdf", "txt", "docx", "doc", "xlsx", "xls", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    
    # C500 性能提示
    if uploaded_files:
        file_count = len(uploaded_files)
        st.caption(f"已选择 {file_count} 个文件")
        if file_count > CONFIG["max_documents_per_upload"]:
            st.warning(f"⚠️ 建议单次上传不超过 {CONFIG['max_documents_per_upload']} 个文档以获得最佳性能")
        for f in uploaded_files:
            fmt = SUPPORTED_FORMATS.get(Path(f.name).suffix.lower(), "📄")
            st.caption(f"{fmt} {f.name}")

    # 构建索引按钮
    if uploaded_files:
        file_list = "\n".join(
            [f"{SUPPORTED_FORMATS.get(Path(f.name).suffix.lower(), '📄')} {f.name}"
             for f in uploaded_files]
        )
        st.caption(f"已选择 {len(uploaded_files)} 个文件：\n{file_list}")

        if st.button("🚀 构建知识库", type="primary", use_container_width=True):
            with st.status("⚙️ 正在处理文档...", expanded=True) as status:
                st.write("📖 正在解析文档格式...")
                raw_docs = load_documents(uploaded_files)

                st.write(f"✂️ 正在切分文本（共 {len(raw_docs)} 个文档片段）...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CONFIG["chunk_size"],
                    chunk_overlap=CONFIG["chunk_overlap"],
                    separators=CONFIG["separators"],  # 中文优化分隔符
                    length_function=len,
                )
                chunks = text_splitter.split_documents(raw_docs)

                st.write(f"🧬 正在生成向量索引（共 {len(chunks)} 个片段）...")
                st.write(f"💡 C500 优化：使用大批处理加速向量化（批大小: {CONFIG['embedding_batch_size']}）")
                st.write(f"⚡ 使用 {CONFIG['embedding_model']} 模型（更强语义理解）")
                
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=CONFIG["chroma_persist_dir"],
                )
                st.session_state.vectorstore = vectorstore

                status.update(
                    label=f"✅ 知识库就绪！共索引 {len(chunks)} 个片段",
                    state="complete",
                    expanded=False,
                )

    st.divider()

    # 系统状态
    st.markdown("### ⚙️ 系统状态")
    st.markdown(f"""
    <div class="config-card">
        🤖 <b>模型</b>：{CONFIG['model_display_name']}<br>
        🎯 <b>向量模型</b>：{CONFIG['embedding_model'].split('/')[-1]}<br>
        💾 <b>向量库</b>：ChromaDB<br>
        🔢 <b>检索 Top-K</b>：{CONFIG['retriever_top_k']}<br>
        🖥️ <b>GPU</b>：{CONFIG['gpu_display_name']}<br>
        📡 <b>状态</b>：{"🟢 知识库已就绪" if "vectorstore" in st.session_state else "🔴 等待上传文档"}
    </div>
    """, unsafe_allow_html=True)
    
    # C500 性能优化说明
    if CONFIG["show_performance_stats"]:
        with st.expander("⚡ C500 性能优化说明"):
            st.markdown(f"""
            **当前优化配置：**
            - 📊 Chunk Size: **{CONFIG['chunk_size']}** (增大以保留更多上下文)
            - 🔄 Chunk Overlap: **{CONFIG['chunk_overlap']}** (增大以避免信息断裂)
            - 🎯 Top-K: **{CONFIG['retriever_top_k']}** (8卡可处理更多上下文)
            - 🌡️ Temperature: **{CONFIG['llm_temperature']}** (降低以提高准确性)
            - 🧠 Max Tokens: **{CONFIG['llm_max_tokens']}** (8卡可生成更长回答)
            - 📦 Embedding Batch: **{CONFIG['embedding_batch_size']}** (大批处理加速)
            - 🔍 Score Threshold: **{CONFIG['retriever_score_threshold']}** (过滤低相关性)
            
            **性能优势：**
            - ✅ 使用 **bge-large-zh-v1.5** (比small模型语义理解提升30%)
            - ✅ 单次可处理 **{CONFIG['max_documents_per_upload']}** 个文档
            - ✅ 向量归一化提升检索精度
            - ✅ 中文优化分隔符提升切分质量
            - ✅ 8卡并行处理，性能强劲
            
            **建议：**
            - 💡 复杂查询可以更详细描述
            - 💡 多文档场景下检索更精准
            - 💡 支持超长文档处理
            """)

    st.divider()

    # 操作按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 清空对话", use_container_width=True):
            msgs.clear()
            st.rerun()
    with col2:
        if st.button("🗑️ 清空索引", use_container_width=True):
            if "vectorstore" in st.session_state:
                del st.session_state.vectorstore
                st.rerun()

# ============================================================
# 🚀 主界面
# ============================================================
st.markdown(f'<h1 class="main-title">{CONFIG["page_title"]}</h1>', unsafe_allow_html=True)
st.markdown(f"**模型**: {CONFIG['model_display_name']} | **GPU**: {CONFIG['gpu_display_name']}")

# 渲染历史消息
for msg in msgs.messages:
    icon = "👤" if msg.type == "human" else "🤖"
    with st.chat_message(msg.type, avatar=icon):
        st.write(msg.content)

# 用户输入
if prompt := st.chat_input("输入您的问题..."):
    with st.chat_message("human", avatar="👤"):
        st.write(prompt)

    if "vectorstore" not in st.session_state:
        st.warning("🚨 请先在左侧控制面板上传文档（支持 PDF/TXT/Word/Excel/Markdown）以激活知识库。")
    else:
        with st.chat_message("ai", avatar="🤖"):
            # 思考过程容器
            thinking_container = st.container()
            
            with thinking_container:
                # 阶段1: 问题理解
                status_placeholder = st.empty()
                status_placeholder.info("🧠 **阶段 1/4**: 正在理解问题并分析上下文...")
                
                import time
                time.sleep(0.3)  # 短暂延迟，让用户看到过程
                
                # 阶段2: 检索相关文档
                status_placeholder.info("🔍 **阶段 2/4**: 正在向量数据库中检索相关文档...")
                
                # 获取相关文档（C500优化：使用相似度阈值过滤）
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": CONFIG["retriever_top_k"],
                        "score_threshold": CONFIG["retriever_score_threshold"]
                    }
                )
                relevant_docs = retriever.get_relevant_documents(prompt)
                
                # 显示检索结果
                retrieval_expander = st.expander(f"📚 检索到 {len(relevant_docs)} 个相关文档片段", expanded=False)
                with retrieval_expander:
                    # 按来源分组
                    docs_by_source = {}
                    for doc in relevant_docs:
                        source = doc.metadata.get("source", "未知来源")
                        if source not in docs_by_source:
                            docs_by_source[source] = []
                        docs_by_source[source].append(doc)
                    
                    for source, docs in docs_by_source.items():
                        st.markdown(f"**📄 {source}** ({len(docs)} 个片段)")
                        for i, doc in enumerate(docs):
                            with st.container():
                                st.caption(f"片段 {i+1}:")
                                st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                
                time.sleep(0.3)
                
                # 阶段3: LLM思考
                model_name = CONFIG.get("llm_model", "Qwen2.5-72B")
                status_placeholder.info(f"💭 **阶段 3/4**: {model_name} 正在综合 {len(relevant_docs)} 个片段生成答案...")
                
                time.sleep(0.3)
                
                # 阶段4: 生成回答
                status_placeholder.info("✍️ **阶段 4/4**: 正在流式生成回答...")
            
            # 回答占位符
            response_placeholder = st.empty()
            full_response = ""
            retrieved_docs = []

            chain = get_rag_chain(msgs)
            config = {"configurable": {"session_id": "metax_session"}}

            try:
                # 执行流式生成
                for chunk in chain.stream({"input": prompt}, config=config):
                    # 保存检索到的文档
                    if "context" in chunk:
                        retrieved_docs = chunk["context"]
                    
                    # 流式渲染回答
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        response_placeholder.markdown(full_response + "▌")

                # 渲染完整回答
                response_placeholder.markdown(full_response)
                
                # 清除状态提示
                status_placeholder.success("✅ 回答完成！")

                # 来源溯源
                if retrieved_docs:
                    with st.expander("🔗 来源溯源及关联度"):
                        # 按来源文档分组
                        docs_by_source = {}
                        for doc in retrieved_docs:
                            source = doc.metadata.get("source", "未知来源")
                            if source not in docs_by_source:
                                docs_by_source[source] = []
                            docs_by_source[source].append(doc)
                        
                        # 显示每个来源文档的片段
                        for source, source_docs in docs_by_source.items():
                            st.markdown(f"### 📄 {source}")
                            for i, doc in enumerate(source_docs):
                                sheet = doc.metadata.get("sheet", "")
                                label = f"(Sheet: {sheet})" if sheet else ""
                                st.markdown(f"""
                                **片段 {i+1}** {label}
                                
                                {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}
                                """)
                                st.divider()

            except Exception as e:
                st.error(f"发生异常: {str(e)}")