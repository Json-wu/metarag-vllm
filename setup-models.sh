#!/bin/bash
# Ollama 模型配置脚本
# 用于在 Docker 容器启动后手动拉取所需模型

set -e

echo "========================================="
echo "  Ollama 模型配置脚本"
echo "========================================="
echo ""

# 检查 Ollama 容器是否运行
if ! docker ps | grep -q ollama-service; then
    echo "❌ 错误：ollama-service 容器未运行"
    echo "请先启动容器：docker-compose up -d"
    exit 1
fi

echo "✅ Ollama 容器运行中"
echo ""

# 等待 Ollama 服务就绪
echo "⏳ 等待 Ollama 服务就绪..."
for i in {1..30}; do
    if docker exec ollama-service ollama list >/dev/null 2>&1; then
        echo "✅ Ollama 服务已就绪"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ 错误：Ollama 服务启动超时"
        exit 1
    fi
    sleep 2
done
echo ""

# 显示当前已安装的模型
echo "📋 当前已安装的模型："
docker exec ollama-service ollama list
echo ""

# 拉取 llama3.2 模型
echo "📥 开始拉取 llama3.2 模型（约 2GB）..."
if docker exec ollama-service ollama pull llama3.2; then
    echo "✅ llama3.2 模型拉取成功"
else
    echo "❌ llama3.2 模型拉取失败"
    exit 1
fi
echo ""

# 拉取 nomic-embed-text 模型
echo "📥 开始拉取 nomic-embed-text 模型（约 274MB）..."
if docker exec ollama-service ollama pull nomic-embed-text; then
    echo "✅ nomic-embed-text 模型拉取成功"
else
    echo "❌ nomic-embed-text 模型拉取失败"
    exit 1
fi
echo ""

# 显示最终安装的模型
echo "========================================="
echo "  ✅ 所有模型配置完成！"
echo "========================================="
echo ""
echo "📋 已安装的模型列表："
docker exec ollama-service ollama list
echo ""
echo "🚀 现在可以访问应用了：http://localhost:8501"
