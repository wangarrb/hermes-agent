# Ollama GPU Offload Thresholds for Local Hindsight

适用：排查“本地 Ollama 模型明明能用 GPU，但长上下文任务为什么突然很慢”。

## 环境样例

- GPU: NVIDIA GeForce RTX 2070 8GB
- 驱动: 580.126.09
- CUDA: 13.0
- Ollama: 0.17.7
- 额外显存占用：Xorg、gnome-shell、gnome-remote-desktop-daemon

## 已验证现象

### 1) 小上下文并非 CPU 回退

对 `qwen3.5:9b-local`，小上下文请求时：
- `ollama ps` 显示 `100% GPU`
- `journalctl -u ollama` 可见：
  - `offloaded 33/33 layers to GPU`
  - `offloading output layer to GPU`
  - `model weights device=CUDA0 size="4.7 GiB"`
  - `model weights device=CPU size="545.6 MiB"`
  - `kv cache device=CUDA0 size="1.2 GiB"`
  - `total memory size="6.5 GiB" ~ "6.6 GiB"`

注意：`100% GPU` 不等于所有权重/内存都只在 GPU；仍可能存在部分 CPU 内存驻留。但对性能判断来说，它表示计算路径处于全 GPU offload 档位。

### 2) 存在清晰的 ctx 阈值，而不是平滑变慢

实测梯度：
- `num_ctx=248` -> `100% GPU`
- `num_ctx=252/255/256` -> `20%/80% CPU/GPU`

这说明机器存在一个很窄的显存临界点；一旦跨过去，推理模式会从全 GPU 切换到混合模式。

### 3) 跨阈值后吞吐会断崖

同模型、同任务、`num_predict=256`：
- `ctx=248`:
  - elapsed ≈ 13.1s
  - eval ≈ 40.5 tok/s
- `ctx=252`:
  - elapsed ≈ 17.2s
  - eval ≈ 20.5 tok/s

结论：只多几个 ctx token，也可能让吞吐近乎腰斩。

## 对 Hindsight 的解释

Hindsight 的不同能力，对上下文压力差异很大：
- retain / recall：通常较短，更容易保持在 GPU 档
- reflect / consolidation：上下文更长，更容易越过阈值，掉到混合推理

因此：
- “retain/recall 没问题，但 reflect/consolidation 特别慢”
- 很多时候不是模型质量问题，也不是服务挂了
- 而是上下文规模把本地 Ollama 推过了显存阈值

## 排查建议

1. 不要只测一次 `ollama ps`
   - 要在实际请求运行期间看，而不是空闲时看。
2. 同时看 `journalctl -u ollama`
   - 重点搜：`offloaded`、`output layer`、`CUDA0`。
3. 用梯度 ctx 实验找阈值
   - 比如 128、192、224、240、248、252、256。
4. 记录显存背景占用
   - `Xorg`、桌面、远程桌面守护进程会显著影响临界点。
5. 优化顺序
   - 先减 ctx / prompt / recall fan-out
   - 再考虑换更小模型或更激进量化
   - 最后才考虑“是不是 Ollama 坏了”

## 默认决策建议

如果目标是“让 Hindsight 本地默认稳定可用”而不是“做极限长上下文实验”：
- retain/recall 可以继续本地化
- reflect/consolidation 不要默认高频开启
- 8GB 卡优先把本地模型定位成低成本、低频、可接受延迟的辅助引擎
