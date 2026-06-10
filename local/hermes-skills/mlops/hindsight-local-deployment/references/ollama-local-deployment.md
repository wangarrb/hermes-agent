# Ollama Local Deployment Reference

Hindsight/Hermes 用本地 Ollama 时需要先部署好模型。本文档记录常见坑和工作流。

## 环境变量冲突

用户 shell 可能设置了：
- `OLLAMA_HOST=https://ollama.com.cn`
- `OLLAMA_MODEL_HOST=https://ollama.com.cn`

这些会让 CLI 偏到远端。本地操作前先 unset：
```bash
unset OLLAMA_HOST OLLAMA_MODEL_HOST
export OLLAMA_HOST=http://127.0.0.1:11434
```

## Systemd 服务代理配置

系统 ollama.service 默认无代理，拉 registry.ollama.ai 会超时。

**方案 A：配置 systemd drop-in（需要 sudo）**
```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/proxy.conf <<'EOF'
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
Environment="NO_PROXY=127.0.0.1,localhost"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**方案 B：用户态 serve（无 sudo）**
```bash
env HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 \
  OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=~/.ollama/models \
  /usr/local/bin/ollama serve &
OLLAMA_HOST=http://127.0.0.1:11435 ollama pull <model>
```

## GGUF 导入

从 HuggingFace 下载 GGUF 后导入 Ollama：

1. **下载**（aria2c + 代理）：
```bash
aria2c --all-proxy=http://127.0.0.1:7890 -x 8 -s 8 -c \
  https://huggingface.co/<repo>/resolve/main/<file>.gguf
```

2. **验证完整性**：
```bash
ls -lh <file>.gguf
# 对比 HuggingFace 页面声明的大小
```

3. **导入**：
```bash
mkdir -p ~/.ollama/models-blobs
cp <file>.gguf ~/.ollama/models-blobs/
cd ~/.ollama
echo 'FROM ./models-blobs/<file>.gguf' > Modelfile
OLLAMA_HOST=http://127.0.0.1:11434 ollama create <name> -f Modelfile
```

4. **验证**：
```bash
OLLAMA_HOST=http://127.0.0.1:11434 ollama list
OLLAMA_HOST=http://127.0.0.1:11434 ollama run <name>
```

## OpenAI 兼容 API

Ollama 提供 OpenAI 兼容端点：
- `http://localhost:11434/v1/chat/completions`
- `http://localhost:11434/v1/models`

Hermes/Hindsight 配置：
```yaml
# Hermes config.yaml
model:
  provider: custom
  base_url: http://127.0.0.1:11434/v1
  default: qwen3.5:9b-local
```

```json
// Hindsight config.json
{
  "llm_provider": "openai",
  "llm_model": "qwen3.5:9b-local",
  "llm_base_url": "http://127.0.0.1:11434/v1",
  "llm_api_key": "ollama"
}
```

## 常见错误

| 错误 | 原因 | 解决 |
|---|---|---|
| `dial tcp: lookup ollama.com.cn: no such host` | OLLAMA_HOST 指向错误域名 | unset 或设为本地地址 |
| `registry.ollama.ai i/o timeout` | systemd 服务无代理 | 配 proxy.conf 或用户态 serve |
| `tensor offset+size exceeds file size` | GGUF 文件不完整 | 重新下载 |

## Qwen3.5-9B Q4_K_M

官方 Ollama tag：`qwen3.5:9b` 或 `qwen3.5:9b-q4_K_M`
HuggingFace GGUF：`unsloth/Qwen3.5-9B-GGUF`
文件大小：约 5.63 GB

直接拉取（代理配置好后）：
```bash
OLLAMA_HOST=http://127.0.0.1:11434 ollama pull qwen3.5:9b
```