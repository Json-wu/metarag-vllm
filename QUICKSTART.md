# 🚀 快速启动指南

## 方式一：Docker 部署（推荐）

### 1. 启动服务

```bash
# 启动所有容器
docker-compose up -d

# 查看容器状态
docker ps
```

### 2. 配置 Ollama 模型

```bash
# 运行模型配置脚本
./setup-models.sh
```

或者手动拉取：

```bash
# 拉取 LLM 模型（约 2GB）
docker exec ollama-service ollama pull llama3.2

# 拉取 Embedding 模型（约 274MB）
docker exec ollama-service ollama pull nomic-embed-text

# 验证模型已安装
docker exec ollama-service ollama list
```

### 3. 访问应用

浏览器打开：`http://localhost:8501`

### 4. 常用命令

```bash
# 查看日志
docker-compose logs -f rag-m4
docker-compose logs -f ollama

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 完全清理（包括数据卷）
docker-compose down -v
```

---

## 方式二：本地运行

### 1. 安装 Ollama

```bash
# macOS
brew install ollama

# 启动服务
brew services start ollama
```

### 2. 拉取模型

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. 安装 Python 依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. 启动应用

```bash
# 设置环境变量（使用本地 Ollama）
export OLLAMA_HOST=http://localhost:11434

# 启动应用
streamlit run rag_m4.py
```

### 5. 访问应用

浏览器打开：`http://localhost:8501`

---

## 📝 使用说明

1. **上传文档**：点击左侧侧边栏的"📥 数据注入"，选择文档
2. **构建索引**：点击"🚀 构建知识库"按钮
3. **开始对话**：在底部输入框输入问题
4. **查看来源**：展开"🔗 来源溯源"查看引用片段

---

## 🐛 常见问题

### 问题：Ollama 连接失败

**解决方案**：

```bash
# Docker 环境
docker exec ollama-service ollama list

# 本地环境
ollama list
```

如果模型未安装，运行 `./setup-models.sh` 或手动拉取模型。

### 问题：应用无法访问

**解决方案**：

```bash
# 检查容器状态
docker ps

# 查看日志
docker-compose logs rag-m4
```

### 问题：文档上传失败

**解决方案**：

- 检查文件格式是否支持（PDF/Word/Excel/TXT/Markdown）
- 查看日志中的错误信息
- 确保文件编码为 UTF-8

---

## 📞 获取帮助

- 查看完整文档：[README.md](./README.md)
- 部署指南：[DEPLOYMENT.md](./DEPLOYMENT.md)
- 提交 Issue：GitHub Issues
