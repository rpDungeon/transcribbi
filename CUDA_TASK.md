# CUDA Backend Task — Voxtral.c

## What This Project Is

Pure C implementation of Mistral AI's **Voxtral Realtime 4B** speech-to-text model. Zero dependencies beyond the C standard library + BLAS. Currently has two backends:
- `make blas` — CPU via OpenBLAS (Linux) or Accelerate (macOS)
- `make mps` — Apple Silicon Metal GPU (macOS only)

We need a **third backend: CUDA** for NVIDIA GPUs.

## Why

On CPU (8-core Ryzen 7 5700X with OpenMP OpenBLAS), transcription runs ~10x slower than real-time:
- 8 seconds of audio takes ~80 seconds (18s encoder + 61s decoder at ~500ms/step)
- This makes the live webapp (browser mic → voxtral → streaming text) unusable

On a 3090, we expect ~10-20ms/step decoder, easily real-time.

## Target Hardware

- **Dual NVIDIA RTX 3090** (24GB VRAM each)
- Only need one GPU — model is ~9GB BF16 weights
- CUDA toolkit should be available in the container

## Architecture Overview

```
WAV → 16kHz → Mel Spectrogram → Conv Stem → Encoder (32 layers) → Downsample 4x → Adapter → Decoder (26 layers) → Tokens
```

- **Encoder**: 32-layer causal transformer, 1280 dim, 32 heads, sliding window 750
- **Decoder**: 26-layer transformer (Ministral-3 based), 3072 dim, GQA (32 heads / 8 KV)
- **Weights**: BF16 (~8.9GB), currently mmap'd from safetensors and converted to F32 on-the-fly per matmul
- **Total params**: ~4B (0.6B encoder + 3.4B decoder)

## Key Files

```
voxtral_kernels.c/.h  — ALL math operations (matmul, attention, norms, activations)
                         This is the primary file to add CUDA support to.
voxtral.c/.h          — Pipeline orchestration, streaming API, KV cache management
voxtral_encoder.c     — Encoder forward pass (calls kernels)
voxtral_decoder.c     — Decoder forward pass (calls kernels)
voxtral_audio.c/.h    — Mel spectrogram (can stay on CPU, it's fast)
voxtral_safetensors.c — Weight loading via mmap
voxtral_tokenizer.c   — Tekken tokenizer (stays on CPU)
main.c                — CLI entry point
Makefile              — Build system
```

## What Needs To Change

### 1. `voxtral_kernels.c` — The Critical File

All compute goes through a small set of functions. These are the hot paths:

```c
// Matrix multiplications (the main bottleneck)
void vox_matmul(float *C, const float *A, const float *B, int M, int K, int N);
void vox_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);
void vox_linear(float *y, const float *x, const float *W, const float *b, int seq_len, int in_dim, int out_dim);

// BF16→F32 conversion + matmul (the real bottleneck on CPU — converts every call)
void vox_linear_bf16(float *y, const float *x, const uint16_t *W_bf16, const float *b, int seq_len, int in_dim, int out_dim);

// Attention
void vox_attention(...);       // Multi-head attention with KV cache
void vox_gqa_attention(...);   // Grouped-query attention (decoder)

// Norms and activations (lighter, but still called many times)
void vox_rmsnorm(float *out, const float *x, const float *weight, int dim, float eps);
void vox_softmax(float *x, int n);
void vox_silu_elementwise(float *x, const float *gate, int n);
void vox_gelu(float *x, int n);
```

### 2. Weight Management

Currently weights are mmap'd as BF16 and converted to F32 per matmul call. For CUDA:
- **Best approach**: Load weights once at startup, convert BF16→FP16 (or keep BF16 on Ampere+), store in GPU memory
- Use `cublasGemmEx` with FP16 inputs / FP32 compute for best 3090 tensor core utilization
- The 3090 supports FP16 tensor cores natively, BF16 only via emulation — so FP16 is preferred

### 3. Memory Buffers

Intermediate activations (encoder/decoder hidden states, KV cache) need to live on GPU. The KV cache is managed in `voxtral.c` — look at `vox_stream_t` struct.

### 4. Makefile

Add a `make cuda` target, similar pattern to `make blas` and `make mps`.

## Suggested Approach

1. **Start minimal**: Just replace `cblas_sgemm` calls with `cublasSgemm` + manage GPU memory for weight matrices and buffers. This alone gets most of the speedup.
2. **Weight pre-conversion**: Upload all BF16 weights to GPU as FP16 at load time (`vox_load()`). Use `cublasGemmEx` with `CUBLAS_COMPUTE_32F` and `CUDA_R_16F` inputs.
3. **Custom kernels for non-BLAS ops**: RMSNorm, SiLU, GELU, softmax, RoPE — write simple CUDA kernels or use cuBLAS where possible.
4. **KV cache on GPU**: The attention functions manage KV cache. Keep it in GPU memory.
5. **Keep mel spectrogram on CPU**: It's fast and only runs once per audio chunk.
6. **Keep tokenizer on CPU**: Trivial cost.

## Existing Patterns to Follow

Look at `voxtral_metal.m` / `voxtral_metal.h` for how the Metal backend was added:
- Separate file for GPU-specific code
- `#ifdef USE_METAL` guards in kernels
- GPU weight cache initialized at load time
- Similar pattern would work: `voxtral_cuda.cu` / `voxtral_cuda.h` with `#ifdef USE_CUDA`

## Build & Test

```bash
# Current build (CPU)
make blas

# Test with sample audio
./voxtral -d voxtral-model -i samples/jfk.wav

# Expected output for jfk.wav (11 seconds):
# "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."

# There's also a webapp in webapp/ that streams browser mic audio to voxtral
cd webapp && bun run server.ts
```

## Development Rules (from CLAUDE.md)

1. No dependencies unless absolutely necessary (CUDA/cuBLAS is necessary here)
2. Standard C + CUDA (.cu files are fine)
3. Test every modification against `samples/jfk.wav` and `samples/test_speech.wav`
4. Keep it simple — don't over-engineer
5. BF16 weights → FP16 on GPU, FP32 computation

## Model Download

```bash
./download_model.sh   # ~8.9GB from HuggingFace
```

Downloads to `./voxtral-model/` containing:
- `consolidated.safetensors` — all weights, BF16
- `tekken.json` — tokenizer
- `params.json` — model config
