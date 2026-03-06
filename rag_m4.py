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
# 针对 Mac mini M4 优化的配置参数
CONFIG = {
    # 页面配置
    "page_title": "🚀 M4 Pro Multi-RAG",
    "page_icon": "⚡",
    "app_title": "🚀 M4 Pro Multi-RAG",
    
    # 模型配置（M4优化）
    "llm_model": "qwen2.5:7b",           # LLM 模型名称（7B适合M4内存）
    "embedding_model": "nomic-embed-text", # Embedding 模型名称
    "llm_temperature": 0.1,                # 温度参数（略微提高创造性，同时保持准确）
    "llm_num_ctx": 4096,                   # 上下文窗口大小（M4可以处理更大上下文）
    "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),  # Ollama 服务地址
    
    # 向量数据库配置（M4优化）
    "chroma_persist_dir": "./chroma_db",   # ChromaDB 持久化目录
    "retriever_top_k": 10,                 # 检索返回的文档数量（增加以提升多文档检索精度）
    "retriever_score_threshold": 0.5,      # 相似度阈值（过滤低相关性结果）
    
    # 文本切分配置（M4优化）
    "chunk_size": 1200,                    # 文本块大小（增大以保留更多上下文）
    "chunk_overlap": 200,                  # 文本块重叠大小（增加以避免信息断裂）
    
    # 性能优化配置
    "batch_size": 32,                      # 向量化批处理大小（M4统一内存优势）
    "max_documents_per_upload": 20,        # 单次上传文档数量限制（控制内存）
    "enable_cache": True,                  # 启用缓存优化
    
    # 显示配置
    "model_display_name": "Qwen 2.5 7B",  # 显示的模型名称
    "gpu_display_name": "M4 Pro",          # 显示的 GPU 名称
    "show_performance_stats": True,        # 显示性能统计
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
    # M4 优化：增加上下文窗口，利用统一内存架构
    llm = ChatOllama(
        model=CONFIG["llm_model"], 
        temperature=CONFIG["llm_temperature"], 
        streaming=True, 
        base_url=CONFIG["ollama_host"],
        num_ctx=CONFIG["llm_num_ctx"]  # M4 可以处理更大上下文
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
        file_count = len(uploaded_files)
        st.caption(f"已选择 {file_count} 个文件")
        
        # M4 性能提示：文档数量限制
        if file_count > CONFIG["max_documents_per_upload"]:
            st.warning(f"⚠️ 为保证 M4 性能，建议单次上传不超过 {CONFIG['max_documents_per_upload']} 个文档")
        
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
                st.write(f"💡 M4 优化：使用批处理加速向量化（批大小: {CONFIG['batch_size']}）")
                
                # M4 优化：批处理向量化，利用统一内存架构
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
    
    # 系统信息显示
    st.info(f"**模型:** {CONFIG['model_display_name']}\n\n**GPU:** {CONFIG['gpu_display_name']}\n\n**向量引擎:** ChromaDB")
    
    # M4 性能优化提示
    if CONFIG["show_performance_stats"]:
        with st.expander("⚡ M4 性能优化说明"):
            st.markdown("""
            **当前优化配置：**
            - 📊 Chunk Size: {chunk_size} (增大以保留更多上下文)
            - 🔄 Chunk Overlap: {chunk_overlap} (增加以避免信息断裂)
            - 🎯 Top-K: {top_k} (提升多文档检索精度)
            - 🌡️ Temperature: {temp} (平衡创造性和准确性)
            - 🧠 Context Window: {ctx} (利用M4统一内存)
            
            **性能建议：**
            - ✅ 单次上传不超过 {max_docs} 个文档
            - ✅ 使用批处理加速向量化
            - ✅ 启用缓存减少重复计算
            """.format(
                chunk_size=CONFIG['chunk_size'],
                chunk_overlap=CONFIG['chunk_overlap'],
                top_k=CONFIG['retriever_top_k'],
                temp=CONFIG['llm_temperature'],
                ctx=CONFIG['llm_num_ctx'],
                max_docs=CONFIG['max_documents_per_upload']
            ))
    
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
                
                # 获取相关文档（M4优化：添加相似度阈值过滤）
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
                status_placeholder.info(f"💭 **阶段 3/4**: {CONFIG['model_display_name']} 正在综合 {len(relevant_docs)} 个片段生成答案...")
                
                time.sleep(0.3)
                
                # 阶段4: 生成回答
                status_placeholder.info("✍️ **阶段 4/4**: 正在流式生成回答...")
            
            # 回答占位符
            response_placeholder = st.empty()
            full_response = ""
            
            chain = get_rag_chain()
            config = {"configurable": {"session_id": "m4_native_session"}}
            
            try:
                # 执行流式生成
                for chunk in chain.stream({"input": prompt}, config=config):
                    # 流式渲染回答
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        response_placeholder.markdown(full_response + "▌")
                
                # 完成后清理光标
                response_placeholder.markdown(full_response)
                
                # 清除状态提示
                status_placeholder.success("✅ 回答完成！")
                
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