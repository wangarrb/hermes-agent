# RTX 2070 8GB local model sizing notes

Purpose: practical sizing guidance for this user's local Ollama + Hermes/Hindsight experiments on an RTX 2070 8GB laptop with desktop VRAM overhead.

## Verified environment
- GPU: NVIDIA RTX 2070 8GB
- Desktop processes consume non-trivial VRAM (`gnome-shell`, remote desktop, NX, etc.)
- Runtime: Ollama local server

## 1) `qwen3.5:9b-local` is too tight for practical full-GPU use

Model metadata:
- family: `qwen35`
- parameter size: `9.0B`
- quantization: `Q4_K_M`
- model max context metadata: `262144`

Measured behavior:
- `num_ctx=128`
  - `size ~= 7.046 GB`
  - `size_vram ~= 7.046 GB`
  - can reach `33/33 layers` + `output layer GPU`
- `num_ctx=256`
  - `size ~= 7.090 GB`
  - `size_vram ~= 5.679 GB`
  - already mixed CPU/GPU
- `num_ctx=4096`
  - `size ~= 7.312 GB`
  - `size_vram ~= 5.897 GB`
  - `32/33 layers`, `output layer CPU`
- `num_ctx=8192`
  - `size ~= 7.455 GB`
  - mixed CPU/GPU
- `num_ctx=16384`
  - `size ~= 7.740 GB`
  - mixed CPU/GPU
- `num_ctx=32768`
  - `size ~= 8.413 GB`
  - further degradation (`29/33 layers` observed)

Interpretation:
- this model can be fully offloaded only in a tiny context window
- for Hermes/Hindsight real prompts, treat it as a mixed CPU/GPU model on this machine
- local `reflect` / `consolidation` being slow is consistent with this hardware/model fit

## 2) Lower 9B quants are possible, but this is a quality-for-VRAM trade

Candidate GGUF sizes from Hugging Face:
- `Qwen_Qwen3.5-9B-Q4_K_M.gguf` — `5.890 GB`
- `Qwen_Qwen3.5-9B-IQ4_XS.gguf` — `5.204 GB`
- `Qwen_Qwen3.5-9B-Q3_K_M.gguf` — `4.868 GB`
- `Qwen_Qwen3.5-9B-Q3_K_S.gguf` — `4.617 GB`
- `Qwen_Qwen3.5-9B-IQ3_XS.gguf` — `4.507 GB`
- `Qwen_Qwen3.5-9B-IQ3_XXS.gguf` — `4.350 GB`
- `Qwen_Qwen3.5-9B-Q2_K.gguf` — `3.998 GB`

Using the observed 4k overhead from the current Q4_K_M run, rough 4k total-memory estimates are:
- 9B `IQ4_XS` -> `~6.626 GB`
- 9B `Q3_K_M` -> `~6.290 GB`
- 9B `Q3_K_S` -> `~6.039 GB`
- 9B `Q2_K` -> `~5.420 GB`

Practical recommendation:
- if user insists on 9B, test in this order:
  1. `IQ4_XS`
  2. `Q3_K_M`
- frame it explicitly as an experiment that may recover 4k pure-GPU at the cost of quality
- do not present it as the default local Hermes/Hindsight solution

## 3) Smaller models are a better fit

Already verified on this machine:

### `qwen2:7b-instruct`
Model metadata:
- parameter size: `7.6B`
- quantization: `Q4_0`
- max context metadata: `32768`

Measured behavior:
- `num_ctx=4096`
  - `size ~= 4.673 GB`
  - `size_vram ~= 4.673 GB`
  - full GPU (`29/29 layers`, `output layer GPU`)
  - `~70 tok/s`
- `num_ctx=8192`
  - `size ~= 5.090 GB`
  - full GPU
  - `~68 tok/s`
- `num_ctx=16384`
  - `size ~= 6.046 GB`
  - full GPU
  - `~69 tok/s`
- `num_ctx=32768`
  - `size ~= 8.429 GB`
  - `size_vram ~= 6.197 GB`
  - degrades to partial offload (`19/29 layers` observed)
  - `~14.6 tok/s`

Interpretation:
- this proves the machine is capable of practical full-GPU local inference
- the bottleneck is not Ollama itself; it is the `9B + 8GB + desktop VRAM overhead` combination

### Qwen3.5-4B sizing notes
Candidate GGUF sizes:
- `Qwen3.5-4B-Q4_K_M.gguf` — `2.741 GB`
- `Qwen3.5-4B-Q5_K_M.gguf` — `3.144 GB`
- `Qwen3.5-4B-Q6_K.gguf` — `3.526 GB`
- `Qwen3.5-4B-Q3_K_M.gguf` — `2.293 GB`

Rough 4k total-memory estimates using the same overhead shape:
- 4B `Q4_K_M` -> `~4.163 GB`
- 4B `Q5_K_M` -> `~4.566 GB`
- 4B `Q6_K` -> `~4.948 GB`

Interpretation:
- `Qwen3.5-4B` is the most plausible long-term local Hermes/Hindsight candidate if the user wants to stay in the Qwen3.5 family
- start with `Q4_K_M`; move to `Q5_K_M` if quality is acceptable and VRAM headroom remains comfortable

## 4) Recommended decision order for this user
1. Fastest validation: switch shadow Hindsight from `qwen3.5:9b-local` to an already-available smaller model (e.g. `qwen2:7b-instruct`) and re-test `retain/recall/reflect/consolidation`
2. Preferred family-preserving path: import `Qwen3.5-4B` and benchmark `4k/8k/16k`
3. Only then, if needed, experiment with lower 9B quants (`IQ4_XS`, `Q3_K_M`)

## Notes
- `num_ctx` in Ollama is in tokens, not K
- do not infer offload mode from model name alone; verify using `/api/ps`, `ollama ps`, and `journalctl -u ollama`
- for long-prompt workloads like Hindsight, prioritize sustained full-GPU at practical context lengths over raw parameter count
