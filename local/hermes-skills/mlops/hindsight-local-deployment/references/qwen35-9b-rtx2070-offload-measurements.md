# Qwen3.5 9B Q4_K_M on RTX 2070 8GB: Ollama offload measurements

Environment:
- GPU: RTX 2070 8GB
- Desktop/remote display processes consume part of VRAM
- Ollama: local service on `127.0.0.1:11434`
- Model: `qwen3.5:9b-local`
- Quantization: `Q4_K_M`

Important correction:
- Ollama `num_ctx` is measured in tokens, not “K”.
- Example: `num_ctx=4096` means 4096 tokens (4k), not 4096k.

Model metadata (`POST /api/show`):
- `parameter_size`: `9.0B`
- `quantization_level`: `Q4_K_M`
- `qwen35.context_length`: `262144`

Measured offload behavior:

| num_ctx | offload result | notes |
|---|---|---|
| 32768 | `29/33 layers to GPU`, `output layer -> CPU` | mixed; slowest in sweep |
| 16384 | `32/33 layers to GPU`, `output layer -> CPU` | mixed |
| 8192  | `32/33 layers to GPU`, `output layer -> CPU` | mixed |
| 4096  | `32/33 layers to GPU`, `output layer -> CPU` | mixed |
| 2048  | `32/33 layers to GPU`, `output layer -> CPU` | mixed |
| 1024  | `32/33 layers to GPU`, `output layer -> CPU` | mixed |
| 512   | `32/33 layers to GPU`, `output layer -> CPU` | mixed |
| 256   | `32/33 layers to GPU`, `output layer -> CPU` | mixed |
| 128   | `33/33 layers to GPU`, `output layer -> GPU` | first pure-GPU point observed |

Representative throughput:
- `num_ctx=4096`: ~18.6 tok/s
- `num_ctx=32768`: ~10.2 tok/s
- `num_ctx=128`: ~40 tok/s

Representative VRAM observations:
- Total VRAM: 8192 MiB
- During mixed runs, `nvidia-smi` showed only ~492–900 MiB free
- Ollama compute app around ~5.7–6.1 GiB, with remaining VRAM consumed by desktop/remote processes

Interpretation:
- This model can be pure GPU only in a very small context window on this machine.
- For realistic Hindsight/Hermes prompts (recall fan-out, reflect, consolidation), pure GPU is not a realistic expectation on RTX 2070 8GB.
- Slow reflect/consolidation is better explained by context-driven mixed CPU/GPU inference than by “Ollama did not use GPU at all”.

Recommended verification commands:
```bash
# model metadata
curl -s http://127.0.0.1:11434/api/show -d '{"model":"qwen3.5:9b-local"}' | jq

# loaded model / active context length
curl -s http://127.0.0.1:11434/api/ps | jq

# offload logs
journalctl -u ollama -n 200 --no-pager | rg 'offloaded|output layer|CUDA0|kv cache|total memory'
```

Use this reference when the user asks:
- "Why is local Qwen so slow?"
- "Can this theoretically run fully on GPU?"
- "Why does Hindsight reflect/consolidation get much slower than simple prompts?"
