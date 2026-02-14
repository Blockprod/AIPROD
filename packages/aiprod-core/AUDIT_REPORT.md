# AIPROD-Core Package — Complete Code Audit Report

**Date:** 2025-07-21  
**Scope:** `packages/aiprod-core/` — every source file  
**Verdict:** The package contains **two distinct, parallel codebases** that do not interact. See § Critical Finding below.

---

## Critical Finding: Dual Codebase

The package ships two **entirely separate** implementations under the same roof:

| Layer | Location | Status | Quality |
|---|---|---|---|
| **Production engine** | `src/aiprod_core/` | Real, functional, production-grade | High — well-structured diffusion transformer with audio+video VAEs, Gemma-3 text encoding, tiled decoding, LoRA, fp8, etc. |
| **Prototype / toy models** | `src/models/`, `src/training/`, `src/data/` | Self-contained toy implementations | Low — hard-coded fake data, no real tokenizer, char-level "multilingual" encoder, no connection to the production engine |

**Nothing in `src/models/`, `src/training/`, or `src/data/` imports from `src/aiprod_core/`.** They are dead code relative to the production engine.

---

## Package Metadata

| Field | Value |
|---|---|
| **pyproject.toml** | 37 lines |
| **Build system** | `uv_build` |
| **Python** | `>=3.10` |
| **Core deps** | `torch ~2.7`, `torchaudio`, `einops`, `transformers ~4.57`, `safetensors`, `accelerate`, `scipy` |
| **Optional deps** | `triton` (Triton kernels), `xformers` (memory-efficient attn), `flash_attn_interface` (FlashAttention 3) |

---

## File-by-File Audit

### 1. Root & Package Init

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 1 | `pyproject.toml` | 37 | Package config with dependencies | N/A | transformers ~4.57, torch ~2.7, torchaudio, einops, safetensors, accelerate, scipy | — |
| 2 | `src/aiprod_core/__init__.py` | 0 | Empty | — | — | — |

### 2. Core Utilities (`src/aiprod_core/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 3 | `tools.py` | ~150 | `VideoLatentTools`, `AudioLatentTools` — factory helpers that wire together patchifier, scheduler, guider, diffusion step, noiser | ✅ Real | torch | — |
| 4 | `types.py` | ~165 | `VideoLatentShape`, `AudioLatentShape`, `LatentState`, `SpatioTemporalScaleFactors` — core data types | ✅ Real | torch | — |
| 5 | `utils.py` | ~60 | `rms_norm`, `to_velocity`, `to_denoised`, `find_matching_file` | ✅ Real | torch | — |

### 3. Components (`src/aiprod_core/components/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 6 | `__init__.py` | ~3 | Docstring only | — | — | — |
| 7 | `diffusion_steps.py` | ~25 | `EulerDiffusionStep` — single Euler diffusion step | ✅ Real | torch | — |
| 8 | `guiders.py` | ~240 | `CFGGuider`, `CFGStarRescalingGuider`, `STGGuider`, `AIPRODAPGGuider`, `LegacyStatefulAPGGuider`, `MultiModalGuider` — classifier-free guidance variants | ✅ Real | torch | — |
| 9 | `noisers.py` | ~37 | `GaussianNoiser` — adds Gaussian noise to latents | ✅ Real | torch | — |
| 10 | `patchifiers.py` | ~350 | `VideoLatentPatchifier`, `AudioPatchifier`, `get_pixel_coords` — converts latents to/from patchified sequence format | ✅ Real | torch, einops | — |
| 11 | `protocols.py` | ~100 | `Patchifier`, `SchedulerProtocol`, `GuiderProtocol`, `DiffusionStepProtocol` — runtime protocols | ✅ Real | torch | — |
| 12 | `schedulers.py` | ~130 | `AIPROD2Scheduler`, `LinearQuadraticScheduler`, `BetaScheduler` — noise schedule generators | ✅ Real | torch, scipy.stats.beta | — |

### 4. Conditioning (`src/aiprod_core/conditioning/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 13 | `__init__.py` | ~5 | Re-exports | — | — | — |
| 14 | `exceptions.py` | ~6 | `ConditioningError` exception class | ✅ Real | — | — |
| 15 | `item.py` | ~20 | `ConditioningItem` protocol | ✅ Real | torch | — |
| 16 | `types/keyframe_cond.py` | ~55 | `VideoConditionByKeyframeIndex` — keyframe-based conditioning | ✅ Real | torch | — |
| 17 | `types/latent_cond.py` | ~50 | `VideoConditionByLatentIndex` — latent-index conditioning | ✅ Real | torch | — |
| 18 | `types/reference_video_cond.py` | ~80 | `VideoConditionByReferenceLatent` — reference-video conditioning (IC-LoRA style) | ✅ Real | torch | — |

### 5. Guidance (`src/aiprod_core/guidance/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 19 | `__init__.py` | ~3 | Re-exports | — | — | — |
| 20 | `perturbations.py` | ~80 | `PerturbationType`, `Perturbation`, `PerturbationConfig`, `BatchedPerturbationConfig` — STG/APG perturbation data structures | ✅ Real | torch | — |

### 6. Loader (`src/aiprod_core/loader/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 21 | `__init__.py` | ~30 | Comprehensive re-exports | — | — | — |
| 22 | `fuse_loras.py` | ~100 | `apply_loras`, `fused_add_round_launch`, `calculate_weight_float8_` — LoRA fusion with optional Triton stochastic rounding | ✅ Real | torch, safetensors, triton (optional) | — |
| 23 | `kernels.py` | ~70 | `fused_add_round_kernel` — Triton kernel for stochastic rounding in fp8 | ✅ Real | triton | — |
| 24 | `module_ops.py` | ~15 | `ModuleOps` NamedTuple (ctor, dtype, device) | ✅ Real | torch | — |
| 25 | `primitives.py` | ~100 | `StateDict`, `StateDictLoader`, `ModelBuilderProtocol`, `LoRAAdaptableProtocol` — loader abstractions | ✅ Real | torch | — |
| 26 | `registry.py` | ~85 | `Registry`, `DummyRegistry`, `StateDictRegistry` — thread-safe state-dict registry | ✅ Real | torch, safetensors | — |
| 27 | `sd_ops.py` | ~130 | `SDOps`, `ContentReplacement`, `ContentMatching` — state-dict key renaming/filtering DSL | ✅ Real | — | — |
| 28 | `sft_loader.py` | ~70 | `SafetensorsStateDictLoader`, `SafetensorsModelStateDictLoader` — safetensors file loader | ✅ Real | safetensors | — |
| 29 | `single_gpu_model_builder.py` | ~110 | `SingleGPUModelBuilder` — builds model on single GPU, supports LoRA fusion | ✅ Real | torch, accelerate | — |

### 7. Model — Common (`src/aiprod_core/model/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 30 | `__init__.py` | ~5 | Re-exports `ModelConfigurator`, `ModelType` | — | — | — |
| 31 | `model_protocol.py` | ~12 | `ModelConfigurator` protocol, `ModelType` TypeVar | ✅ Real | — | — |
| 32 | `common/normalization.py` | ~60 | `NormType`, `PixelNorm`, `build_normalization_layer` | ✅ Real | torch | — |

### 8. Model — Audio VAE (`src/aiprod_core/model/audio_vae/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 33 | `__init__.py` | ~25 | Comprehensive exports | — | — | — |
| 34 | `audio_vae.py` | 481 | `AudioEncoder`, `AudioDecoder`, `decode_audio` — full audio VAE with mel-spectrogram encode/decode | ✅ Real | torch, torchaudio | — |
| 35 | `attention.py` | ~75 | `AttnBlock`, `AttentionType` enum — self-attention for audio VAE | ✅ Real | torch | — |
| 36 | `causality_axis.py` | ~12 | `CausalityAxis` enum (TIME, FREQUENCY) | ✅ Real | — | — |
| 37 | `causal_conv_2d.py` | ~120 | `CausalConv2d`, `make_conv2d` — causal 2D convolutions for spectrograms | ✅ Real | torch | — |
| 38 | `downsample.py` | ~130 | `Downsample`, `build_downsampling_path` — progressive audio downsampling | ✅ Real | torch | — |
| 39 | `model_configurator.py` | ~130 | `VocoderConfigurator`, `AudioDecoderConfigurator`, `AudioEncoderConfigurator` + SDOps key maps | ✅ Real | torch | — |
| 40 | `ops.py` | ~80 | `AudioProcessor` (mel spectrogram computation), `PerChannelStatistics` | ✅ Real | torch, torchaudio | — |
| 41 | `resnet.py` | ~200 | `ResBlock1`, `ResBlock2`, `ResnetBlock` — audio ResNet blocks (HiFi-GAN style) | ✅ Real | torch | — |
| 42 | `upsample.py` | ~110 | `Upsample`, `build_upsampling_path` — progressive audio upsampling | ✅ Real | torch | — |
| 43 | `vocoder.py` | ~130 | `Vocoder` — HiFi-GAN style vocoder for waveform generation | ✅ Real | torch | — |

### 9. Model — Upsampler (`src/aiprod_core/model/upsampler/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 44 | `__init__.py` | ~10 | Re-exports | — | — | — |
| 45 | `model.py` | ~160 | `LatentUpsampler`, `upsample_video` — upsamples latents before decoding | ✅ Real | torch | — |
| 46 | `model_configurator.py` | ~35 | `LatentUpsamplerConfigurator` + SDOps | ✅ Real | — | — |
| 47 | `blur_downsample.py` | ~65 | `BlurDownsample` — binomial-kernel anti-aliased downsampling | ✅ Real | torch | — |
| 48 | `pixel_shuffle.py` | ~60 | `PixelShuffleND` — 1D/2D/3D pixel-shuffle (sub-pixel convolution) | ✅ Real | torch, einops | — |
| 49 | `res_block.py` | ~40 | `ResBlock` — residual block for upsampler | ✅ Real | torch | — |
| 50 | `spatial_rational_resampler.py` | ~55 | `SpatialRationalResampler` — fractional spatial resampling | ✅ Real | torch | — |

### 10. Model — Transformer (`src/aiprod_core/model/transformer/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 51 | `__init__.py` | ~15 | Re-exports | — | — | — |
| 52 | `adaln.py` | ~40 | `AdaLayerNormSingle` — adaptive layer norm (PixArt-Alpha style) | ✅ Real | torch | PixArt-Alpha pattern (in design) |
| 53 | `attention.py` | ~200 | `PytorchAttention`, `XFormersAttention`, `FlashAttention3`, `Attention` — multi-backend attention with RoPE | ✅ Real | torch, xformers (opt), flash_attn_interface (opt) | — |
| 54 | `feed_forward.py` | ~15 | `FeedForward` with GELUApprox activation | ✅ Real | torch | — |
| 55 | `gelu_approx.py` | ~10 | `GELUApprox` | ✅ Real | torch | — |
| 56 | `modality.py` | ~20 | `Modality` dataclass (latent, timesteps, cross-attn context, mask, coords) | ✅ Real | torch | — |
| 57 | `model.py` | 469 | `AIPRODModel` (main diffusion transformer), `AIPRODModelType`, `LegacyX0Model`, `X0Model` — the core denoising model | ✅ Real | torch | — |
| 58 | `model_configurator.py` | ~250 | `AIPRODModelConfigurator`, `AIPRODVideoOnlyModelConfigurator`, fp8 upcast/rounding utilities, SDOps key maps | ✅ Real | torch | — |
| 59 | `rope.py` | 205 | `AIPRODRopeType`, `apply_rotary_emb` (interleaved/split), `precompute_freqs_cis` — rotary position embeddings | ✅ Real | torch | — |
| 60 | `text_projection.py` | ~30 | `PixArtAlphaTextProjection` — text embedding projection layer | ✅ Real | torch | PixArt-Alpha (in class name) |
| 61 | `timestep_embedding.py` | ~130 | `TimestepEmbedding`, `Timesteps`, `PixArtAlphaCombinedTimestepSizeEmbeddings` — sinusoidal timestep embeddings | ✅ Real | torch | PixArt-Alpha (in class name) |
| 62 | `transformer.py` | ~370 | `TransformerConfig`, `BasicAVTransformerBlock` — full audio-video cross-attention transformer block with AdaLN, perturbations | ✅ Real | torch | — |
| 63 | `transformer_args.py` | ~270 | `TransformerArgs`, `TransformerArgsPreprocessor`, `MultiModalTransformerArgsPreprocessor` — pre-processes modality inputs to transformer format | ✅ Real | torch | — |

### 11. Model — Video VAE (`src/aiprod_core/model/video_vae/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 64 | `__init__.py` | ~15 | Re-exports | — | — | — |
| 65 | `video_vae.py` | 927 | `VideoEncoder`, `VideoDecoder`, `decode_video`, tiled decoding with spatial/temporal overlap blending | ✅ Real | torch, einops | — |
| 66 | `model_configurator.py` | ~80 | `VideoEncoderConfigurator`, `VideoDecoderConfigurator` + SDOps key maps | ✅ Real | — | — |
| 67 | `convolution.py` | 318 | `make_conv_nd`, `CausalConv3d`, `DualConv3d` — causal 3D convolutions decomposed into 2D spatial + 1D temporal | ✅ Real | torch, torch.nn.functional, einops | — |
| 68 | `enums.py` | ~25 | `NormLayerType`, `LogVarianceType`, `PaddingModeType` | ✅ Real | — | — |
| 69 | `normalization.py` | 3 | Re-exports from `model.common.normalization` | — | — | — |
| 70 | `ops.py` | ~90 | `patchify`, `unpatchify`, `PerChannelStatistics` — 5D video patchification | ✅ Real | torch, einops | — |
| 71 | `resnet.py` | ~270 | `ResnetBlock3D`, `UNetMidBlock3D` — 3D ResNet blocks with optional attention, noise injection, timestep conditioning | ✅ Real | torch | — |
| 72 | `sampling.py` | ~140 | `SpaceToDepthDownsample`, `DepthToSpaceUpsample` — pixel-shuffle based up/down sampling for video | ✅ Real | torch | — |
| 73 | `tiling.py` | ~370 | `TilingConfig`, `SpatialTilingConfig`, `TemporalTilingConfig`, `Tile`, `create_tiles` — tiled processing infrastructure with trapezoidal blending masks | ✅ Real | torch | — |

### 12. Text Encoders — Gemma (`src/aiprod_core/text_encoders/gemma/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 74 | `text_encoders/__init__.py` | 1 | Docstring only | — | — | — |
| 75 | `gemma/__init__.py` | ~20 | Comprehensive exports of all Gemma components | — | — | — |
| 76 | `config.py` | ~75 | `Gemma3ConfigData`, **`GEMMA3_CONFIG_FOR_AIPROD`** — hardcoded Gemma 3 architecture config | ✅ Real | transformers (Gemma3Config) | **Gemma 3** (hidden_size=3840, num_hidden_layers=48, vocab_size=262208, intermediate_size=24576) |
| 77 | `embeddings_connector.py` | 211 | `Embeddings1DConnector`, `_BasicTransformerBlock1D` — 1D transformer connector with RoPE and learnable registers | ✅ Real | torch | — |
| 78 | `feature_extractor.py` | ~40 | `GemmaFeaturesExtractorProjLinear` — projects Gemma hidden states (3840×49 → 3840) | ✅ Real | torch | — |
| 79 | `tokenizer.py` | ~75 | `AIPRODVGemmaTokenizer` — wraps HuggingFace AutoTokenizer with system-prompt template | ✅ Real | transformers (AutoTokenizer) | — |

### 13. Text Encoders — Gemma Encoders (`src/aiprod_core/text_encoders/gemma/encoders/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 80 | `base_encoder.py` | 310 | `GemmaTextEncoderModelBase` — base class using `Gemma3ForConditionalGeneration`, text preprocessing with system prompts | ✅ Real | torch, transformers (Gemma3ForConditionalGeneration, AutoImageProcessor, Gemma3Processor) | **Gemma 3** |
| 81 | `av_encoder.py` | ~170 | `AVGemmaTextEncoderModel`, `AVGemmaTextEncoderModelConfigurator`, AV_GEMMA_TEXT_ENCODER_KEY_OPS + GEMMA_MODEL_OPS key maps | ✅ Real | torch, transformers (Gemma3Config) | **Gemma 3** (via GEMMA3_CONFIG_FOR_AIPROD) |
| 82 | `video_only_encoder.py` | ~95 | `VideoGemmaTextEncoderModel`, `VideoGemmaTextEncoderModelConfigurator`, VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS | ✅ Real | torch, transformers (Gemma3ForConditionalGeneration) | **Gemma 3** |

### 14. Tests (`tests/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 83 | `conftest.py` | 56 | Pytest fixtures and custom markers (slow, integration, gpu) | ✅ Real | pytest, torch | — |
| 84 | `README.md` | 98 | Describes test structure (unit/, integration/, fixtures/) — **but these directories do NOT exist** | N/A | — | — |

---

## ⚠️ Prototype / Toy Code (NOT connected to production engine)

These files live under `src/models/`, `src/training/`, and `src/data/`. They import **only from each other** and **never from `aiprod_core`**. They are self-contained prototypes.

### 15. Toy Models (`src/models/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 85 | `__init__.py` | 13 | Exports `HybridBackbone`, `VideoVAE`, `MultilingualTextEncoder` | — | — | — |
| 86 | `backbone.py` | ~230 | `HybridBackbone` — toy Transformer+CNN hybrid (768-D, 48 layers). Self-contained, no connection to `aiprod_core` transformer. Has `__main__` test block. | ⚠️ Prototype | torch | — |
| 87 | `text_encoder.py` | 308 | `MultilingualTextEncoder` — **char-level** "multilingual" encoder (ASCII bytes → 256-dim embedding → 4 transformer layers → 768-D). Not a real multilingual model. `CrossModalAttention` class. | ⚠️ Prototype | torch | `google/mt5-small` (referenced in docstring/param but **never loaded**) |
| 88 | `vae.py` | ~260 | `VideoVAE` — toy 3D conv VAE (Encoder3D, TemporalAttentionBlock, Decoder3D). Separate from `aiprod_core`'s 927-line production VideoVAE. | ⚠️ Prototype | torch | — |

### 16. Toy Training (`src/training/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 89 | `__init__.py` | 17 | Exports `CurriculumTrainer`, `TrainingPhase`, `CurriculumConfig` | — | — | — |
| 90 | `curriculum.py` | 549 | `CurriculumTrainer`, `CurriculumConfig`, `TrainingPhase` — 5-phase curriculum trainer with thermal monitoring, GPU throttling. Monkey-patches `torch._dynamo` to avoid Windows bugs. | ⚠️ Prototype | torch, subprocess, json | — |
| 91 | `train.py` | ~240 | `Phase1MLTraining` — orchestrates training of toy models. Imports from `src/models/` and `src/training/`. CLI entry point. | ⚠️ Prototype | torch, argparse | — |

### 17. Toy Data (`src/data/`)

| # | File | Lines | Summary | Real? | External Imports | Hardcoded Models |
|---|---|---|---|---|---|---|
| 92 | `__init__.py` | 255 | `CurriculumVideoDataset`, `VideoDataLoader` — generates **synthetic video frames** (colored rectangles on gray backgrounds). No real video loading. | ⚠️ Prototype | torch, numpy | — |

---

## Summary Statistics

| Category | Files | Total Lines (approx) |
|---|---|---|
| Production engine (`aiprod_core/`) | 73 | ~7,500 |
| Prototype/toy code (`models/`, `training/`, `data/`) | 8 | ~1,870 |
| Tests | 2 | ~154 |
| Config/metadata | 1 | 37 |
| **Total** | **84** | **~9,560** |

---

## External Dependencies Summary

### Production (`aiprod_core`)
| Library | Usage |
|---|---|
| `torch ~2.7` | Everywhere — core tensor ops, nn.Module, autograd |
| `torchaudio` | Audio VAE: mel spectrogram, InverseMelScale, GriffinLim, resampling |
| `einops` | Rearrange operations in patchifiers, video VAE, pixel shuffle |
| `transformers ~4.57` | `Gemma3ForConditionalGeneration`, `Gemma3Config`, `Gemma3Processor`, `AutoTokenizer`, `AutoImageProcessor` |
| `safetensors` | Model weight loading (SafetensorsStateDictLoader) |
| `accelerate` | `init_empty_weights` for model building |
| `scipy` | `scipy.stats.beta` for BetaScheduler |
| `triton` | Custom GPU kernel `fused_add_round_kernel` for stochastic rounding |
| `xformers` | Optional memory-efficient attention backend |
| `flash_attn_interface` | Optional FlashAttention 3 backend |

### Prototype (`models/`, `training/`, `data/`)
| Library | Usage |
|---|---|
| `torch` | Everything |
| `numpy` | Synthetic data generation |
| `subprocess` | nvidia-smi GPU clock limiting |

---

## Hardcoded Model References

| Reference | File(s) | Details |
|---|---|---|
| **Google Gemma 3** | `config.py`, `base_encoder.py`, `av_encoder.py`, `video_only_encoder.py` | `GEMMA3_CONFIG_FOR_AIPROD`: hidden_size=3840, num_hidden_layers=48, vocab_size=262208, intermediate_size=24576, num_attention_heads=32, num_key_value_heads=16 |
| **PixArt-Alpha** (pattern) | `adaln.py`, `text_projection.py`, `timestep_embedding.py` | Class names reference PixArt-Alpha architecture patterns: `AdaLayerNormSingle`, `PixArtAlphaTextProjection`, `PixArtAlphaCombinedTimestepSizeEmbeddings` |
| `google/mt5-small` | `text_encoder.py` (prototype) | Referenced in docstring/default param but **never actually loaded** |

---

## Key Observations

1. **No actual tests exist.** `tests/README.md` describes `unit/`, `integration/`, `fixtures/` directories that don't exist. Only `conftest.py` with fixtures is present.

2. **The prototype layer is dead weight.** `src/models/backbone.py` (HybridBackbone), `src/models/text_encoder.py` (char-level encoder), `src/models/vae.py` (toy VAE), `src/training/`, and `src/data/` are never imported by the production engine and have no integration path.

3. **The production engine is well-architected.** Clean separation of concerns: loader (weight loading/LoRA), model (transformer, VAEs, upsampler), components (schedulers, guiders, patchifiers), conditioning, text encoding. Protocol-based design enables swapping implementations.

4. **Tiled decoding is sophisticated.** The video VAE supports spatial and temporal tiling with trapezoidal blending masks for seamless chunk boundaries — a production-quality feature for memory-constrained GPUs.

5. **Multi-modal support is complete.** The transformer handles audio+video jointly via `BasicAVTransformerBlock` with separate modality-specific norms, cross-attention, and output projections.

6. **The `curriculum.py` monkey-patches `torch._dynamo`** by injecting fake modules into `sys.modules` to avoid Windows frame inspection bugs. This is fragile and version-dependent.

7. **fp8 support** is implemented via Triton kernels (`fused_add_round_kernel`) and `calculate_weight_float8_` for quantized LoRA fusion.
