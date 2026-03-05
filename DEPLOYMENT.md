# 🚀 MetaRAG-vLLM 部署指南

本文档提供 MetaRAG-vLLM 系统在不同环境下的详细部署说明，包括：
- Mac mini M4 本地部署
- 沐曦 MetaX C500 GPU 集群部署
- Docker 容器化部署
- 云服务器部署（阿里云/腾讯云等）

---

## 📑 目录

1. [Mac mini M4 本地部署](#1-mac-mini-m4-本地部署)
2. [沐曦 MetaX C500 GPU 集群部署](#2-沐曦-metax-c500-gpu-集群部署)
3. [Docker 容器化部署](#3-docker-容器化部署)
4. [云服务器部署](#4-云服务器部署)
5. [性能调优](#5-性能调优)
6. [故障排查](#6-故障排查)

---

## 1. Mac mini M4 本地部署

### 1.1 系统要求

- **硬件**：Mac mini M4 或其他 Apple Silicon Mac
- **操作系统**：macOS 14.0 (Sonoma) 或更高版本
- **内存**：建议 16GB 及以上
- **存储**：至少 20GB 可用空间

### 1.2 安装 Homebrew（如未安装）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 1.3 安装 Python 3.9+

```bash
# 检查 Python 版本
python3 --version

# 如果版本低于 3.9，使用 Homebrew 安装
brew install python@3.11
```

### 1.4 安装 Ollama

```bash
# 使用 Homebrew 安装
brew install ollama

# 或者从官网下载安装包
# https://ollama.ai/download
```

### 1.5 启动 Ollama 服务

```bash
# 方式 1：前台运行（用于调试）
ollama serve

# 方式 2：后台运行（推荐）
brew services start ollama
```

### 1.6 拉取所需模型

```bash
# 拉取 LLM 模型（约 2GB）
ollama pull llama3.2

# 拉取 Embedding 模型（约 274MB）
ollama pull nomic-embed-text

# 验证模型已安装
ollama list
```

### 1.7 克隆项目并安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/metarag-vllm.git
cd metarag-vllm

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.8 启动应用

```bash
# 启动 Mac M4 版本
streamlit run rag_m4.py

# 应用将在 http://localhost:8501 启动
```

### 1.9 验证部署

1. 浏览器访问 `http://localhost:8501`
2. 上传测试文档（如 PDF 或 TXT 文件）
3. 点击"🚀 构建知识库"
4. 输入问题测试对话功能

---

## 2. 沐曦 MetaX C500 GPU 集群部署

### 2.1 系统要求

- **硬件**：沐曦 MetaX C500 GPU × 8
- **操作系统**：Ubuntu 20.04/22.04 LTS
- **CUDA**：兼容沐曦驱动的 CUDA 版本
- **内存**：建议 128GB 及以上
- **存储**：至少 500GB SSD（用于模型和向量库）

### 2.2 安装系统依赖

```bash
# 更新系统
sudo apt-get update && sudo apt-get upgrade -y

# 安装基础工具
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    htop \
    nvidia-utils-535  # 根据实际驱动版本调整
```

### 2.3 安装 Python 3.9+

```bash
# Ubuntu 20.04
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev

# Ubuntu 22.04（自带 Python 3.10+）
sudo apt-get install -y python3 python3-venv python3-pip
```

### 2.4 安装 vLLM

```bash
# 创建虚拟环境
python3 -m venv vllm-env
source vllm-env/bin/activate

# 安装 vLLM（根据 CUDA 版本选择）
pip install vllm

# 或从源码安装（如需自定义）
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 2.5 下载 Qwen2.5-72B 模型

```bash
# 安装 Hugging Face CLI
pip install huggingface_hub

# 登录 Hugging Face（需要 token）
huggingface-cli login

# 下载模型（约 145GB）
huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
    --local-dir /data/models/Qwen2.5-72B-Instruct \
    --local-dir-use-symlinks False
```

### 2.6 启动 vLLM 服务

```bash
# 创建启动脚本
cat > start_vllm.sh << 'EOF'
#!/bin/bash

python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code
EOF

chmod +x start_vllm.sh

# 后台启动 vLLM
nohup ./start_vllm.sh > vllm.log 2>&1 &

# 查看日志
tail -f vllm.log
```

### 2.7 下载 Embedding 模型

```bash
# 下载 bge-small-zh-v1.5（约 400MB）
huggingface-cli download BAAI/bge-small-zh-v1.5 \
    --local-dir /data/models/bge-small-zh-v1.5 \
    --local-dir-use-symlinks False
```

### 2.8 部署 RAG 应用

```bash
# 克隆项目
cd /opt
git clone https://github.com/yourusername/metarag-vllm.git
cd metarag-vllm

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
pip install langchain-openai langchain-huggingface
```

### 2.9 配置应用

编辑 `rag_metax.py` 配置：

```bash
vim rag_metax.py
```

修改以下配置：

```python
CONFIG = {
    "llm_base_url": "http://localhost:8000/v1",  # vLLM 服务地址
    "llm_model": "Qwen2.5-72B-Instruct",
    "embedding_model": "/data/models/bge-small-zh-v1.5",  # 本地模型路径
    "embedding_device": "cuda",
    "chroma_persist_dir": "/data/chroma_db",  # 向量库持久化路径
}
```

### 2.10 启动应用

```bash
# 前台启动（测试）
streamlit run rag_metax.py --server.port=8501 --server.address=0.0.0.0

# 后台启动（生产）
nohup streamlit run rag_metax.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    > streamlit.log 2>&1 &
```

### 2.11 配置系统服务（可选）

创建 systemd 服务文件：

```bash
sudo vim /etc/systemd/system/metarag.service
```

内容如下：

```ini
[Unit]
Description=MetaRAG vLLM Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/metarag-vllm
Environment="PATH=/opt/metarag-vllm/venv/bin"
ExecStart=/opt/metarag-vllm/venv/bin/streamlit run rag_metax.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable metarag
sudo systemctl start metarag
sudo systemctl status metarag
```

---

## 3. Docker 容器化部署

### 3.1 前置要求

- Docker 20.10+
- Docker Compose 1.29+
- 足够的磁盘空间（至少 10GB）

### 3.2 安装 Docker 和 Docker Compose

**Ubuntu/Debian：**

```bash
# 更新系统
sudo apt-get update

# 安装 Docker
curl -fsSL https://get.docker.com | bash -s docker

# 启动 Docker 服务
sudo systemctl start docker
sudo systemctl enable docker

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker --version
docker-compose --version
```

**macOS：**

```bash
# 使用 Homebrew 安装
brew install docker docker-compose

# 或下载 Docker Desktop
# https://www.docker.com/products/docker-desktop
```

### 3.3 构建镜像

```bash
# 进入项目目录
cd metarag-vllm

# 构建镜像
docker-compose build

# 查看镜像
docker images | grep rag-m4
```

### 3.4 启动服务

```bash
# 启动服务（后台运行）
docker-compose up -d

# 查看运行状态
docker-compose ps

# 查看日志
docker-compose logs -f rag-m4
```

### 3.5 访问应用

浏览器访问：`http://localhost:8501`

### 3.6 常用 Docker 命令

```bash
# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看资源使用
docker stats

# 进入容器调试
docker exec -it rag-m4-app bash

# 清理未使用的资源
docker system prune -a
```

### 3.7 配置 Ollama 服务（可选）

如果需要在 Docker 中运行 Ollama：

```bash
# 启动 Ollama 容器
docker-compose up -d ollama

# 进入容器拉取模型
docker exec -it ollama-service bash
ollama pull llama3.2
ollama pull nomic-embed-text
exit

# 修改 rag_m4.py 中的 Ollama 地址
# llm = ChatOllama(model="llama3.2", base_url="http://ollama:11434")
```

---

## 4. 云服务器部署

### 4.1 阿里云 ECS 部署

#### 4.1.1 服务器配置推荐

| 配置项 | Mac M4 版本 | MetaX C500 版本 |
|--------|------------|----------------|
| **CPU** | 4核 | 32核+ |
| **内存** | 8GB | 128GB+ |
| **GPU** | 无 | MetaX C500 × 8 |
| **存储** | 50GB SSD | 500GB SSD |
| **带宽** | 5Mbps | 100Mbps |

#### 4.1.2 安全组配置

在阿里云控制台配置安全组规则：

| 规则方向 | 协议类型 | 端口范围 | 授权对象 | 说明 |
|---------|---------|---------|---------|------|
| 入方向 | TCP | 8501 | 0.0.0.0/0 | Streamlit UI |
| 入方向 | TCP | 8000 | 127.0.0.1/32 | vLLM API（内网） |
| 入方向 | TCP | 22 | 你的IP | SSH 登录 |

#### 4.1.3 部署步骤

```bash
# 1. SSH 登录服务器
ssh root@your-server-ip

# 2. 上传项目文件
# 方式 1：使用 Git
git clone https://github.com/yourusername/metarag-vllm.git
cd metarag-vllm

# 方式 2：使用 SCP
# 本地执行：scp -r metarag-vllm root@your-server-ip:/opt/

# 3. 按照前面的步骤部署
# Mac M4 版本：参考第 1 节
# MetaX C500 版本：参考第 2 节
# Docker 版本：参考第 3 节
```

### 4.2 腾讯云 CVM 部署

部署步骤与阿里云类似，主要差异：

1. **安全组配置**：在腾讯云控制台 → 云服务器 → 安全组
2. **防火墙规则**：开放 8501 端口
3. **其他步骤**：与阿里云相同

### 4.3 华为云 ECS 部署

部署步骤与阿里云类似，主要差异：

1. **安全组配置**：在华为云控制台 → 弹性云服务器 → 安全组
2. **其他步骤**：与阿里云相同

### 4.4 配置 Nginx 反向代理（推荐）

#### 4.4.1 安装 Nginx

```bash
sudo apt-get install -y nginx
```

#### 4.4.2 配置 Nginx

创建配置文件：

```bash
sudo vim /etc/nginx/sites-available/metarag
```

内容如下：

```nginx
server {
    listen 80;
    server_name your-domain.com;  # 替换为你的域名

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 支持
        proxy_read_timeout 86400;
    }
}
```

启用配置：

```bash
sudo ln -s /etc/nginx/sites-available/metarag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 4.4.3 配置 HTTPS（使用 Let's Encrypt）

```bash
# 安装 Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com

# 自动续期测试
sudo certbot renew --dry-run
```

---

## 5. 性能调优

### 5.1 Mac M4 优化

#### 5.1.1 模型优化

```bash
# 使用量化模型减少内存占用
ollama pull llama3.2:7b-q4_0  # 4-bit 量化版本
```

#### 5.1.2 参数调优

编辑 `rag_m4.py`：

```python
# 减少检索数量
search_kwargs={"k": 3}  # 从 5 降到 3

# 减少 chunk 大小
chunk_size=600
chunk_overlap=50

# 降低温度
temperature=0
```

### 5.2 MetaX C500 优化

#### 5.2.1 vLLM 优化

```bash
# 启动脚本优化
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.95 \  # 提高 GPU 利用率
    --max-model-len 16384 \           # 增加上下文长度
    --enable-chunked-prefill \        # 启用分块预填充
    --max-num-batched-tokens 8192 \   # 批处理优化
    --port 8000
```

#### 5.2.2 Embedding 优化

```python
# 使用更大的 Embedding 模型
CONFIG = {
    "embedding_model": "BAAI/bge-large-zh-v1.5",  # 从 small 升级到 large
    "embedding_device": "cuda",
}
```

#### 5.2.3 向量库优化

```python
# 增加检索数量和重排序
retriever = vectorstore.as_retriever(
    search_type="mmr",  # 使用 MMR 算法
    search_kwargs={
        "k": 10,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)
```

### 5.3 系统级优化

#### 5.3.1 增加文件描述符限制

```bash
# 临时设置
ulimit -n 65535

# 永久设置
echo "* soft nofile 65535" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65535" | sudo tee -a /etc/security/limits.conf
```

#### 5.3.2 配置 Swap（如内存不足）

```bash
# 创建 16GB swap
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久挂载
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 6. 故障排查

### 6.1 常见问题

#### 问题 1：应用无法访问

**症状**：浏览器无法打开 `http://localhost:8501`

**排查步骤**：

```bash
# 1. 检查进程是否运行
ps aux | grep streamlit

# 2. 检查端口占用
netstat -tulpn | grep 8501
lsof -i :8501

# 3. 查看日志
tail -f streamlit.log

# 4. 检查防火墙
sudo ufw status
sudo ufw allow 8501/tcp
```

#### 问题 2：Ollama 连接失败

**症状**：`Connection refused` 或 `Ollama not found`

**解决方案**：

```bash
# 检查 Ollama 服务状态
brew services list | grep ollama

# 重启 Ollama
brew services restart ollama

# 检查端口
curl http://localhost:11434/api/tags
```

#### 问题 3：vLLM 内存不足

**症状**：`CUDA out of memory`

**解决方案**：

```bash
# 降低 GPU 内存利用率
--gpu-memory-utilization 0.8

# 减少最大序列长度
--max-model-len 4096

# 使用量化模型
--quantization awq  # 或 gptq
```

#### 问题 4：向量库加载失败

**症状**：`ChromaDB not found` 或 `Permission denied`

**解决方案**：

```bash
# 检查目录权限
ls -la chroma_db/
sudo chown -R $USER:$USER chroma_db/

# 清理并重建
rm -rf chroma_db/
# 重新上传文档构建索引
```

#### 问题 5：文档解析失败

**症状**：`Failed to load document`

**解决方案**：

```bash
# 检查依赖是否完整
pip install pypdf docx2txt openpyxl python-docx unstructured

# 检查文件编码
file -I your-document.txt

# 尝试转换编码
iconv -f GBK -t UTF-8 input.txt > output.txt
```

### 6.2 日志查看

```bash
# Streamlit 日志
tail -f streamlit.log

# vLLM 日志
tail -f vllm.log

# Docker 日志
docker-compose logs -f

# 系统日志
sudo journalctl -u metarag -f
```

### 6.3 性能监控

```bash
# GPU 使用情况
nvidia-smi -l 1

# 系统资源
htop

# Docker 资源
docker stats

# 磁盘使用
df -h
du -sh chroma_db/
```

---

## 7. 备份与恢复

### 7.1 备份向量数据库

```bash
# 备份 ChromaDB
tar -czf chroma_db_backup_$(date +%Y%m%d).tar.gz chroma_db/

# 上传到云存储（示例：阿里云 OSS）
ossutil cp chroma_db_backup_*.tar.gz oss://your-bucket/backups/
```

### 7.2 恢复向量数据库

```bash
# 下载备份
ossutil cp oss://your-bucket/backups/chroma_db_backup_20240305.tar.gz .

# 解压恢复
tar -xzf chroma_db_backup_20240305.tar.gz
```

---

## 8. 安全加固

### 8.1 配置访问控制

```bash
# 使用 Nginx 添加基本认证
sudo apt-get install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin

# 修改 Nginx 配置
location / {
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8501;
}
```

### 8.2 配置 IP 白名单

```nginx
# 在 Nginx 配置中添加
location / {
    allow 192.168.1.0/24;  # 允许内网
    allow your-office-ip;   # 允许办公室 IP
    deny all;               # 拒绝其他
    proxy_pass http://localhost:8501;
}
```

---

## 9. 监控告警

### 9.1 使用 Prometheus + Grafana

```bash
# 安装 Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz
tar -xzf prometheus-2.40.0.linux-amd64.tar.gz
cd prometheus-2.40.0.linux-amd64
./prometheus --config.file=prometheus.yml

# 安装 Grafana
sudo apt-get install -y grafana
sudo systemctl start grafana-server
```

### 9.2 配置告警规则

创建 `alert_rules.yml`：

```yaml
groups:
  - name: metarag_alerts
    interval: 30s
    rules:
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
      
      - alert: ServiceDown
        expr: up{job="metarag"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "MetaRAG service is down"
```

---

## 10. 附录

### 10.1 环境变量配置

创建 `.env` 文件：

```bash
# vLLM 配置
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen2.5-72B-Instruct

# Embedding 配置
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cuda

# ChromaDB 配置
CHROMA_PERSIST_DIR=/data/chroma_db

# Streamlit 配置
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### 10.2 资源清单

| 资源 | 大小 | 下载地址 |
|------|------|---------|
| Llama3.2 | ~2GB | `ollama pull llama3.2` |
| nomic-embed-text | ~274MB | `ollama pull nomic-embed-text` |
| Qwen2.5-72B | ~145GB | Hugging Face |
| bge-small-zh-v1.5 | ~400MB | Hugging Face |
| bge-large-zh-v1.5 | ~1.3GB | Hugging Face |

### 10.3 参考链接

- [LangChain 文档](https://python.langchain.com/)
- [Streamlit 文档](https://docs.streamlit.io/)
- [vLLM 文档](https://docs.vllm.ai/)
- [Ollama 文档](https://github.com/ollama/ollama)
- [ChromaDB 文档](https://docs.trychroma.com/)

---

## 📞 技术支持

如有问题，请检查：
1. Docker 和 Docker Compose 版本
2. 服务器资源是否充足
3. 网络和防火墙配置
4. 应用日志输出

如需进一步帮助，请提交 GitHub Issue 或联系技术支持团队。

---

**最后更新时间**：2024-03-05
