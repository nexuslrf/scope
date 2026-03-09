# Helios scope pipeline — dev notes

Comparison between `scope/test.py` (our integration) and the upstream
distilled inference reference (`Helios/scripts/inference/helios-distilled_i2v.sh`
+ `Helios/infer_helios.py` + `pipeline_helios_diffusers.py`).

---

## 1. Sample mode: T2V only vs I2V / V2V

| | Upstream | Scope |
|---|---|---|
| Modes | `--sample_type {t2v,i2v,v2v}` | T2V only |
| Image conditioning | `--image_path` → VAE-encode → `image_latents` + `fake_image_latents` + noise injection | Not implemented |
| Video conditioning | `--video_path` → chunk-encode all frames → noisy `video_latents` prefills history | Not implemented |
| I2V anchor frame | Encoded from provided image | First generated chunk's frame-0 latent |

The i2v shell script specifically sets `--sample_type i2v --image_path example/wave.jpg`.
Our scope pipeline always starts from zeros for `image_latents` (T2V behaviour).

---

## 2. Triton kernel replacements — omitted in scope

`infer_helios.py` (when not using `--enable_compile`) monkey-patches the
transformer with three Triton-based kernels
(`helios/modules/helios_kernels/`):

```
replace_rmsnorm_with_fp32(transformer)
replace_all_norms_with_flash_norms(transformer)
replace_rope_with_flash_rope()
```

### What each one does

**`replace_rmsnorm_with_fp32`** — `fp32_rmsnorm.py`
Patches every `RMSNorm` / `torch.nn.RMSNorm` forward to upcast inputs to
`float32`, compute the norm, then recast to original dtype. Guards against
bfloat16 precision loss in norm statistics.

**`replace_all_norms_with_flash_norms`** — `triton_norm.py`
Replaces `LayerNorm` and `RMSNorm` with fused Triton kernels that compute the
entire normalization in a single GPU pass in float32 internally. Same numerical
result as fp32 replacement above but faster via kernel fusion.

**`replace_rope_with_flash_rope`** — `triton_rope.py`
Monkey-patches the module-level `apply_rotary_emb_transposed` function in both
`transformer_helios_diffusers` and `transformer_helios` with a Triton kernel
(`Flash_RoPE_Transposed`) that processes cos/sin application in a single fused
pass. The test shows zero error vs the PyTorch reference at float32; at bf16
the kernel produces the same results.

**Why scope omits them**: `helios/modules/helios_kernels` is not vendored —
only the diffusers-format transformer and scheduler are. The omission means
norms run in native bfloat16 (potential accumulated rounding) and RoPE uses
the unfused PyTorch reference path. Generation is still correct; there may be
a small quality / numerical difference versus the reference at bfloat16.

---

## 3. Prompt tokenization: max_length 512 vs 226

| | Upstream | Scope |
|---|---|---|
| `max_sequence_length` | 512 (default in `__call__`) | 226 (hard-coded via `HuggingfaceTokenizer(seq_len=226)`) |
| Tokenizer | `AutoTokenizer` loaded with the model | `HuggingfaceTokenizer` wrapping the same tokenizer |
| Prompt cleaning | `prompt_clean()`: ftfy fix + html unescape + whitespace | None |
| Padding style | Same — `padding="max_length"`, zero-fill after actual sequence | Same |

The upstream `_get_t5_prompt_embeds` method has `max_sequence_length: int = 226`
as its own default, but the outer `__call__` passes `max_sequence_length=512`.
For the Helios-Distilled model the cross-attention window over 512 tokens vs
226 tokens likely makes little practical difference (the prompt for `stage2`
distilled generation is typically short), but longer prompts lose their tail
when truncated at 226 in scope.

The missing `prompt_clean()` means HTML entities or multi-space prompts may not
be cleaned before encoding.

---

## 4. Negative prompt / CFG

At `guidance_scale = 1.0` (distilled default), `do_classifier_free_guidance =
False` in both pipelines — the negative prompt embedding is never forwarded
through the transformer. The upstream reference passes a detailed negative
prompt string; scope uses an empty string. **No impact on output** at
guidance_scale=1.0, but scope wastes one text-encoder forward pass per prompt
change (encoding the empty negative string unnecessarily).

---

## 5. Seed strategy per chunk

| | Upstream | Scope |
|---|---|---|
| Generator | `torch.Generator("cuda").manual_seed(args.seed)` — **same seed** for every chunk | `base_seed + total_generated_latent_frames` — **incremented** per chunk |

Upstream reuses the same `generator` object across all chunks; PyTorch
advances its state automatically with each `randn_tensor` call, so chunks
effectively get different noise despite sharing a single generator.
Scope creates a fresh generator per chunk seeded deterministically.
Both approaches give reproducible, varied noise per chunk — the outputs will
differ because the exact noise sequence is different.

---

## 6. History latents buffer

| | Upstream | Scope |
|---|---|---|
| Accumulation | Unbounded: `history_latents = cat([history_latents, latents], dim=2)` across all chunks | Capped immediately at `sum(history_sizes)=19` frames |
| Transformer input | Both slice `history_latents[:, :, -num_history_latent_frames:]` → same 19 frames | Same |
| VAE decode source | `real_history_latents[:, :, -num_latent_frames_per_chunk:]` (from full buffer) | `latents` directly (identical to last N frames) |

The accumulation difference is irrelevant for transformer quality — both feed
the same 19-frame window. It only mattered historically for computing
`real_history_latents` for decoding; scope skips that indirection since
`latents` is already the current chunk.

---

## 7. `is_skip_first_chunk` — not implemented

The upstream pipeline supports `--is_skip_first_chunk`. When enabled, the
first chunk is denoised but discarded; `image_latents` is captured from chunk 1
instead of chunk 0. This can improve the anchor frame quality by letting the
model "warm up". Scope does not implement this and always captures frame-0 of
the very first chunk as the anchor.

---

## 8. Compute optimisations not present in scope

| Feature | Upstream | Scope |
|---|---|---|
| `torch.compile` | `--enable_compile` (text encoder + VAE + transformer) | Not applied |
| Multi-GPU (Context Parallel) | `--enable_parallelism` via `ContextParallelConfig(ulysses_degree=N)` | Single-GPU only |
| Low-VRAM group offload | `--low_vram_mode` via `pipe.enable_group_offload(...)` | Not implemented |

---

## 9. Output format and streaming model

| | Upstream | Scope |
|---|---|---|
| Generation unit | Full video in one call (all chunks in a loop inside `__call__`) | One chunk per `__call__` invocation |
| Output | `HeliosPipelineOutput.frames[0]` — numpy array, HWC float [0,1] | `dict["video"]` — THWC torch tensor [0,1] |
| Streaming | Caller must wait for complete video | Each call immediately returns a decodable chunk |

The streaming design is intentional for scope's real-time web interface —
each chunk is decoded and returned independently so the client can display
frames as they are generated.

---

## Summary

| # | Difference | Quality impact | Priority |
|---|---|---|---|
| 1 | T2V only (no I2V/V2V) | Functional gap | Medium — extend if I2V needed |
| 2 | Triton norms/RoPE not applied | Minor numerical drift at bf16 | Low — fp32 norms would help |
| 3 | max_length 226 vs 512 | Truncates long prompts | Low for typical short prompts |
| 4 | No prompt cleaning | Rare edge case | Low |
| 5 | Negative prompt encoded unnecessarily | ~1 extra T5 forward per prompt change | Low |
| 6 | Seed scheme differs | Determinism differs, not a correctness issue | Informational |
| 7 | `is_skip_first_chunk` absent | First-chunk anchor quality | Low |
| 8 | No compile / multi-GPU / offload | Speed / VRAM only | Medium for production |
