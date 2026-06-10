# BAAI/bge-m3 manual download for Hindsight embeddings

Use this when Hindsight embedding/model download is unstable and the user chooses to download manually.

## Safety / workflow

1. Stop any background model download before giving manual links, so the user does not race a partial writer or waste bandwidth.
2. If Hindsight is actively using the embedding model, stop or pause the service/container before replacing files.
3. Give direct file URLs, sizes, and hashes for the minimal PyTorch/SentenceTransformer set; do not ask the user to download the whole repository.
4. After the user provides the local model directory, restart Hindsight and verify both:
   - the model loaded is `BAAI/bge-m3` / local bge-m3 path
   - DB embedding dimension matches bge-m3: `memory_units.embedding = vector(1024)`

## Minimal files for `SentenceTransformer("BAAI/bge-m3")`

Total minimal set from the 2026-05-10 check:

- 2,295,419,991 bytes
- 2.14 GiB
- 2.30 GB

Required files:

- `pytorch_model.bin` — 2,271,145,830 B
  sha256: `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`
  URL: `https://huggingface.co/BAAI/bge-m3/resolve/main/pytorch_model.bin`

- `tokenizer.json` — 17,098,108 B
  sha256: `21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08`
  URL: `https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json`

- `sentencepiece.bpe.model` — 5,069,051 B
  sha256: `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865`
  URL: `https://huggingface.co/BAAI/bge-m3/resolve/main/sentencepiece.bpe.model`

- `colbert_linear.pt` — 2,100,674 B
  sha256: `19bfbae397c2b7524158c919d0e9b19393c5639d098f0a66932c91ed8f5f9abb`
  URL: `https://huggingface.co/BAAI/bge-m3/resolve/main/colbert_linear.pt`

- `sparse_linear.pt` — 3,516 B
  sha256: `45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9`
  URL: `https://huggingface.co/BAAI/bge-m3/resolve/main/sparse_linear.pt`

Small required config/tokenizer files:

- `config.json`
- `config_sentence_transformers.json`
- `modules.json`
- `1_Pooling/config.json`
- `sentence_bert_config.json`
- `special_tokens_map.json`
- `tokenizer_config.json`

Direct URL pattern for small files:

`https://huggingface.co/BAAI/bge-m3/resolve/main/<path>`

Example for nested pooling config:

`https://huggingface.co/BAAI/bge-m3/resolve/main/1_Pooling/config.json`

## Avoid downloading

Do not download `onnx/` for normal Hindsight PyTorch/SentenceTransformer use.

Approximate sizes from the same check:

- minimal PyTorch set: 2.14 GiB / 2.30 GB
- ONNX set alone: 2.13 GiB / 2.29 GB
- full repository with ONNX and extras: 4.27 GiB / 4.59 GB

## Container cache path pitfall

The Hindsight Docker image runs as `/home/hindsight`, so Hugging Face defaults to `/home/hindsight/.cache/huggingface`, not the host user's `$HOME/.cache/huggingface` path.

If the container only mounts:

- `$HOME/.cache/huggingface:$HOME/.cache/huggingface`

then `SentenceTransformer("BAAI/bge-m3")` can ignore the prepared host cache and start downloading into the container writable layer, typically visible as:

- `/home/hindsight/.cache/huggingface/hub/models--BAAI--bge-m3/blobs/<sha>.incomplete`

Fix the long-term container recreation script/env, not just the live container:

- mount `$HOME/.cache/huggingface:/home/hindsight/.cache/huggingface`
- mount `$HOME/.cache/torch:/home/hindsight/.cache/torch`
- set `HF_HOME=/home/hindsight/.cache/huggingface`
- set `HUGGINGFACE_HUB_CACHE=/home/hindsight/.cache/huggingface/hub`
- optionally set `TRANSFORMERS_CACHE=/home/hindsight/.cache/huggingface/hub`

Verification signal:

- container logs show `Embeddings: local provider initialized (dim: 1024)` without creating new `.incomplete` blobs under `/home/hindsight/.cache/huggingface`.

## Verification reminders

After placing files, prefer a real load test over just checking file presence. Then run a small recall smoke/audit after production retain finishes, because wrong embedding dimensions can appear only at DB insert/search time.