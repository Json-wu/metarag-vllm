# 🚀 MetaRAG-vLLM

基于国产GPU的高性能RAG知识库助手系统，支持本地Mac mini M4和国产沐曦C500 GPU集群部署。

## 📖 项目简介

MetaRAG-vLLM 是一个功能完整的检索增强生成（RAG）知识库问答系统，专为国产GPU优化设计。系统支持多种文档格式索引、多轮对话记忆、实时思考链展示，提供企业级的文档智能问答能力。

### ✨ 核心特性

- **🎯 双平台支持**
  - Mac mini M4：本地开发测试，使用 Ollama + Llama3.2
  - 沐曦 MetaX C500：生产环境部署，支持 vLLM + Qwen2.5-72B

- **📚 多格式文档支持**
  - PDF 文档（学术论文、技术手册）
  - Word 文档（.docx, .doc）
  - Excel 表格（.xlsx, .xls，保留表格结构）
  - 纯文本（.txt）
  - Markdown 文档（.md）

- **🧠 智能对话能力**
  - 多轮对话上下文记忆
  - 基于历史的问题改写
  - 语义向量检索（Top-K）
  - 来源溯源与引用展示

- **💡 思考链可视化**
  - 实时展示检索过程
  - 动态显示语义片段数量
  - 流式输出生成过程
  - 透明化AI推理步骤

- **🎨 现代化UI**
  - 基于 Streamlit 的交互界面
  - 深色主题侧边栏
  - 渐变色标题特效
  - 动画思考卡片

## 🏗️ 技术架构

### 技术栈

| 组件 | Mac M4 版本 | MetaX C500 版本 |
|------|------------|----------------|
| **LLM** | Llama3.2 (Ollama) | Qwen2.5-72B (vLLM) |
| **Embedding** | nomic-embed-text | bge-small-zh-v1.5 |
| **向量数据库** | ChromaDB | ChromaDB |
| **框架** | LangChain + Streamlit | LangChain + Streamlit |
| **文档解析** | PyPDF, Docx2txt, Pandas | PyPDF, Docx2txt, Pandas |

### 系统架构图

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│  (文档上传 + 对话界面 + 思考链展示 + 来源溯源)              │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                  LangChain RAG Chain                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 问题改写      │→│ 向量检索      │→│ 答案生成      │  │
│  │ (多轮上下文)  │  │ (Top-K)      │  │ (流式输出)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌───────▼────────┐
│  ChromaDB      │  │  LLM Service   │
│  向量索引       │  │  Ollama/vLLM   │
│  (持久化存储)   │  │  (模型推理)     │
└────────────────┘  └────────────────┘
```

## 🚀 快速开始

### 前置要求

**Mac M4 版本：**
- macOS 14.0+
- Python 3.9+
- Ollama（用于运行 Llama3.2 和 nomic-embed-text）

**MetaX C500 版本：**
- Ubuntu 20.04/22.04
- Python 3.9+
- CUDA 兼容驱动
- vLLM 服务（运行 Qwen2.5-72B）

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/yourusername/metarag-vllm.git
cd metarag-vllm
```

#### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置模型服务

**Mac M4 - 使用 Ollama：**

```bash
# 安装 Ollama
brew install ollama

# 启动 Ollama 服务
ollama serve

# 拉取模型（新终端）
ollama pull llama3.2
ollama pull nomic-embed-text
```

**MetaX C500 - 使用 vLLM：**

```bash
# 启动 vLLM 服务（示例）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --port 8000
```

修改 `rag_metax.py` 中的配置：

```python
CONFIG = {
    "llm_base_url": "http://localhost:8000/v1",  # vLLM 服务地址
    "llm_model": "Qwen2.5-72B-Instruct",
    # ... 其他配置
}
```

#### 5. 启动应用

**Mac M4 版本：**

```bash
streamlit run rag_m4.py
```

**MetaX C500 版本：**

```bash
streamlit run rag_metax.py
```

访问 `http://localhost:8501` 即可使用。

### Docker 快速部署

如果使用 Docker 部署，请按以下步骤操作：

```bash
# 1. 启动容器
docker-compose up -d

# 2. 配置 Ollama 模型（首次运行必须）
./setup-models.sh

# 或手动拉取模型
docker exec ollama-service ollama pull llama3.2
docker exec ollama-service ollama pull nomic-embed-text

# 3. 访问应用
# 浏览器打开 http://localhost:8501
```

详细说明请参考 [QUICKSTART.md](./QUICKSTART.md)

## 📝 使用指南

### 基本使用流程

1. **上传文档**
   - 点击左侧侧边栏的"📥 数据注入"
   - 选择一个或多个文档（支持 PDF/Word/Excel/TXT/Markdown）
   - 点击"🚀 构建知识库"按钮

2. **等待索引构建**
   - 系统会自动解析文档
   - 切分文本为语义片段
   - 生成向量索引并持久化

3. **开始对话**
   - 在底部输入框输入问题
   - 观察思考链展示检索过程
   - 查看流式生成的答案
   - 展开"🔗 来源溯源"查看引用片段

4. **多轮对话**
   - 系统会自动记忆对话历史
   - 支持上下文关联的追问
   - 问题会自动改写为独立查询

### 高级功能

- **清空对话**：点击侧边栏"🧹 清空对话"按钮
- **清空索引**：点击侧边栏"🗑️ 清空索引"按钮重新构建知识库
- **查看系统状态**：侧边栏实时显示模型、GPU、索引状态

## 🐳 Docker 部署

详细的 Docker 部署指南请参考 [DEPLOYMENT.md](./DEPLOYMENT.md)

### 快速启动

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f rag-m4
```

访问 `http://your-server-ip:8501`

## 📊 性能优化

### Mac M4 优化建议

- 使用量化模型减少内存占用
- 调整 `chunk_size` 和 `chunk_overlap` 参数
- 限制 `retriever_top_k` 数量（建议 3-5）

### MetaX C500 优化建议

- 使用 vLLM 的 Tensor Parallelism 充分利用 8 卡
- 启用 Flash Attention 加速推理
- 调整 `max_tokens` 控制输出长度
- 使用 bge-large-zh 提升向量质量

## 🔧 配置说明

### rag_m4.py 配置

```python
# 模型配置
llm = ChatOllama(model="llama3.2", temperature=0, streaming=True)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 文本切分
chunk_size=900
chunk_overlap=100

# 检索配置
search_kwargs={"k": 5}
```

### rag_metax.py 配置

```python
CONFIG = {
    "llm_base_url": "http://localhost:8000/v1",
    "llm_model": "Qwen2.5-72B-Instruct",
    "llm_temperature": 0.3,
    "llm_max_tokens": 2048,
    
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "embedding_device": "cuda",
    
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "retriever_top_k": 5,
}
```

## 📁 项目结构

```
metarag-vllm/
├── rag_m4.py              # Mac M4 版本主程序
├── rag_metax.py           # MetaX C500 版本主程序
├── requirements.txt       # Python 依赖
├── Dockerfile             # Docker 镜像构建文件
├── docker-compose.yml     # Docker Compose 配置
├── setup-models.sh        # Ollama 模型配置脚本
├── QUICKSTART.md          # 快速启动指南
├── DEPLOYMENT.md          # 详细部署文档
├── README.md              # 项目说明文档
├── chroma_db/             # 向量数据库持久化目录
└── venv/                  # Python 虚拟环境
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - RAG 框架
- [Streamlit](https://streamlit.io/) - Web UI 框架
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [Ollama](https://ollama.ai/) - 本地模型运行
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎
- [Qwen](https://github.com/QwenLM/Qwen) - 通义千问大模型

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至：wyqctt@gmail.com

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
