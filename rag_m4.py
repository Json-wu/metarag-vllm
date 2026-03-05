import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

# ==================== 配置中心 ====================
CONFIG = {
    # 页面配置
    "page_title": "🚀 M4 Pro Multi-RAG",
    "page_icon": "⚡",
    "app_title": "🚀 M4 Pro Multi-RAG",
    
    # 模型配置
    "llm_model": "qwen2.5:7b",           # LLM 模型名称
    "embedding_model": "nomic-embed-text", # Embedding 模型名称
    "llm_temperature": 0,                  # 温度参数
    "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),  # Ollama 服务地址
    
    # 向量数据库配置
    "chroma_persist_dir": "./chroma_db",   # ChromaDB 持久化目录
    "retriever_top_k": 5,                  # 检索返回的文档数量
    
    # 文本切分配置
    "chunk_size": 900,                     # 文本块大小
    "chunk_overlap": 100,                  # 文本块重叠大小
    
    # 显示配置
    "model_display_name": "Qwen 2.5 7B",  # 显示的模型名称
    "gpu_display_name": "M4 Pro",          # 显示的 GPU 名称
}
# ================================================

# --- 1. 页面配置与 UI 深度美化 ---
st.set_page_config(
    page_title=CONFIG["page_title"], 
    page_icon=CONFIG["page_icon"], 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定义 CSS
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
    </style>
""", unsafe_allow_html=True)

# --- 2. 模型初始化 (优化 M4 资源调用) ---
@st.cache_resource
def init_models():
    # 本地环境使用 localhost，Docker 环境通过环境变量 OLLAMA_HOST 覆盖
    llm = ChatOllama(
        model=CONFIG["llm_model"], 
        temperature=CONFIG["llm_temperature"], 
        streaming=True, 
        base_url=CONFIG["ollama_host"]
    )
    embeddings = OllamaEmbeddings(
        model=CONFIG["embedding_model"], 
        base_url=CONFIG["ollama_host"]
    )
    return llm, embeddings

llm, embeddings = init_models()

# --- 3. 记忆管理 ---
msgs = StreamlitChatMessageHistory(key="chat_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("👋 你好！我是您的本地知识助手。准备好深度分析您的文档了吗？")

# --- 4. 侧边栏：Dashboard 风格设计 ---
with st.sidebar:
    st.markdown("## 🛠️ 控制面板")
    st.divider()
    
    # 文件上传区美化
    st.markdown("### 📥 数据注入")
    uploaded_files = st.file_uploader(
        "支持格式: PDF, TXT, Word, Excel, Markdown", 
        type=["pdf", "txt", "docx", "doc", "xlsx", "xls", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    # 显示已上传文件列表
    if uploaded_files:
        st.caption(f"已选择 {len(uploaded_files)} 个文件")
        for file in uploaded_files:
            st.caption(f"📄 {file.name}")
    
    # 构建知识库按钮
    if uploaded_files and st.button("🚀 构建知识库", type="primary", use_container_width=True):
        with st.status("🛸 正在解析并构建向量空间...", expanded=True) as status:
            all_documents = []
            success_count = 0
            failed_files = []
            
            # 处理每个上传的文件
            for uploaded_file in uploaded_files:
                try:
                    st.write(f"📖 正在处理: {uploaded_file.name}")
                    
                    # 获取文件扩展名
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    temp_file = f"temp_{uploaded_file.name}"
                    
                    # 保存临时文件
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # 根据文件类型选择加载器
                    if file_extension == "pdf":
                        loader = PyPDFLoader(temp_file)
                    elif file_extension == "txt":
                        loader = TextLoader(temp_file, encoding="utf-8")
                    elif file_extension in ["docx", "doc"]:
                        loader = Docx2txtLoader(temp_file)
                    elif file_extension in ["xlsx", "xls"]:
                        loader = UnstructuredExcelLoader(temp_file, mode="elements")
                    elif file_extension == "md":
                        loader = UnstructuredMarkdownLoader(temp_file)
                    else:
                        st.warning(f"⚠️ 跳过不支持的文件格式: {uploaded_file.name}")
                        failed_files.append(uploaded_file.name)
                        continue
                    
                    # 加载文档
                    documents = loader.load()
                    
                    # 为每个文档添加来源元数据
                    for doc in documents:
                        doc.metadata["source"] = uploaded_file.name
                    
                    all_documents.extend(documents)
                    success_count += 1
                    st.write(f"✅ {uploaded_file.name} 加载成功 ({len(documents)} 个片段)")
                    
                    # 清理临时文件
                    import os
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name} 加载失败: {str(e)}")
                    failed_files.append(uploaded_file.name)
            
            # 如果有成功加载的文档，构建向量库
            if all_documents:
                st.write(f"✂️ 正在切分文本（共 {len(all_documents)} 个文档片段）...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CONFIG["chunk_size"], 
                    chunk_overlap=CONFIG["chunk_overlap"]
                )
                chunks = text_splitter.split_documents(all_documents)
                
                st.write(f"🧬 正在生成向量索引（共 {len(chunks)} 个语义片段）...")
                vectorstore = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings,
                    persist_directory=CONFIG["chroma_persist_dir"]
                )
                st.session_state.vectorstore = vectorstore
                
                # 保存已加载的文件列表
                if "loaded_files" not in st.session_state:
                    st.session_state.loaded_files = []
                st.session_state.loaded_files = [f.name for f in uploaded_files if f.name not in failed_files]
                
                status.update(label="✅ 知识库构建完成", state="complete", expanded=False)
                st.success(f"🎉 成功加载 {success_count}/{len(uploaded_files)} 个文档，共 {len(chunks)} 个语义片段")
                
                if failed_files:
                    st.warning(f"⚠️ 以下文件加载失败: {', '.join(failed_files)}")
            else:
                status.update(label="❌ 没有成功加载的文档", state="error", expanded=False)
                st.error("所有文档加载失败，请检查文件格式")

    st.divider()
    st.markdown("### ⚙️ 系统状态")
    
    # 显示知识库状态
    if "vectorstore" in st.session_state and "loaded_files" in st.session_state:
        st.success(f"✅ 知识库已就绪")
        st.caption(f"📚 已加载 {len(st.session_state.loaded_files)} 个文档：")
        for filename in st.session_state.loaded_files:
            st.caption(f"  • {filename}")
    else:
        st.warning("⚠️ 知识库未初始化")
    
    st.info(f"**模型:** {CONFIG['model_display_name']}\n\n**GPU:** {CONFIG['gpu_display_name']}\n\n**向量引擎:** ChromaDB")
    
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
            if "loaded_files" in st.session_state:
                del st.session_state.loaded_files
            st.rerun()

# --- 5. RAG 逻辑构建 ---
def get_rag_chain():
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": CONFIG["retriever_top_k"]}
    )
    
    # 问题重写 Prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "根据历史记录将问题重写为独立查询。若已独立，原样返回。仅输出结果。"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # 最终回答 Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个基于专业文档的智能助手。请根据【上下文】提供精准回答。不确定时请直言。不要乱编。\n\n【上下文】：\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# --- 6. 聊天主界面 ---
st.markdown(f'<h1 class="main-title">{CONFIG["app_title"]}</h1>', unsafe_allow_html=True)
st.markdown(f"**模型**: {CONFIG['model_display_name']} | **GPU**: {CONFIG['gpu_display_name']}")

# 渲染历史
for msg in msgs.messages:
    icon = "👤" if msg.type == "human" else "🤖"
    with st.chat_message(msg.type, avatar=icon):
        st.write(msg.content)

# 输入处理
if prompt := st.chat_input("输入您的问题..."):
    # 显示人类问题
    with st.chat_message("human", avatar="👤"):
        st.write(prompt)
    
    if "vectorstore" not in st.session_state:
        st.warning("🚨 请先在左侧控制面板上传文档（支持 PDF/TXT/Word/Excel/Markdown）以激活知识库。")
    else:
        with st.chat_message("ai", avatar="🤖"):
            # A. 动态思考卡片
            thought_placeholder = st.empty()
            with thought_placeholder.container():
                st.markdown("""
                    <div class="thought-card">
                        🔍 <b>Deep Analysis</b>: 正在多轮历史中解析意图并检索向量片段...
                    </div>
                """, unsafe_allow_html=True)
            
            # B. 回答占位符
            response_placeholder = st.empty()
            full_response = ""
            
            chain = get_rag_chain()
            config = {"configurable": {"session_id": "m4_native_session"}}
            
            try:
                # C. 执行流式流
                has_shown_docs = False
                for chunk in chain.stream({"input": prompt}, config=config):
                    # 检索到文档后立即更新思考状态
                    if "context" in chunk and not has_shown_docs:
                        with thought_placeholder.container():
                            st.markdown(f"""
                                <div class="thought-card">
                                    ✅ <b>Context Found</b>: 检索到 {len(chunk['context'])} 个相关语义片段。正在合成回答...
                                </div>
                            """, unsafe_allow_html=True)
                        has_shown_docs = True
                    
                    # 流式渲染回答
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        response_placeholder.markdown(full_response + "▌")
                
                # 完成后清理光标和思考卡片（或保留卡片作为溯源）
                response_placeholder.markdown(full_response)
                
                # D. 增加来源标注（支持多文档溯源）
                with st.expander("🔗 来源溯源及关联度"):
                    docs = st.session_state.vectorstore.as_retriever().get_relevant_documents(prompt)
                    
                    # 按来源文档分组
                    docs_by_source = {}
                    for doc in docs:
                        source = doc.metadata.get("source", "未知来源")
                        if source not in docs_by_source:
                            docs_by_source[source] = []
                        docs_by_source[source].append(doc)
                    
                    # 显示每个来源文档的片段
                    for source, source_docs in docs_by_source.items():
                        st.markdown(f"### 📄 {source}")
                        for i, d in enumerate(source_docs):
                            page = d.metadata.get("page", "N/A")
                            st.markdown(f"""
                            **片段 {i+1}** {f'(第 {page} 页)' if page != 'N/A' else ''}
                            
                            {d.page_content[:300]}{'...' if len(d.page_content) > 300 else ''}
                            """)
                            st.divider()
            
            except Exception as e:
                st.error(f"发生异常: {str(e)}")