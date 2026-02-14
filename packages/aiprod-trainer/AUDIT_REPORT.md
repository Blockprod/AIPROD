# AIPROD-Trainer Package — Comprehensive Code Audit Report

**Date:** 2026-02-11  
**Scope:** Every file in `packages/aiprod-trainer/`  
**Auditor:** Automated (GitHub Copilot)

---

## Executive Summary

The `aiprod-trainer` package is a **fully functional fine-tuning toolkit** for the **AIPROD** (Lightricks LTX-based) audio-video generation model. It is **NOT** training a proprietary model from scratch — it fine-tunes an existing open-source diffusion model (transformer-based, flow-matching) using LoRA adapters or full fine-tuning.

**Key findings:**
- **100% real implementation** — No stubs, no placeholder code, no TODO-only files
- **Heavy dependency on `aiprod-core`** — All model architectures (transformer, VAE, text encoder, scheduler, patchifier) come from the sibling `aiprod-core` package
- **Hardcoded external model references:** Qwen/Qwen2.5-Omni-7B, Gemma 3, Gemini Flash, AIPROD/LTX-Video
- **Training paradigm:** Fine-tuning via LoRA (PEFT) on precomputed latents using flow matching
- **Total source files:** 42 files (16 library modules, 8 scripts, 3 YAML configs, 4 accelerate configs, 10 test files, 1 template)

---

## File-by-File Audit

### A. Core Library — `src/aiprod_trainer/`

---

#### 1. `src/aiprod_trainer/__init__.py`
- **Lines:** 43
- **Summary:** Package initializer. Configures Python logging with Rich markup handler. Detects multi-GPU rank via `LOCAL_RANK` env var and suppresses logs on non-zero ranks.
- **Implementation:** ✅ Real, functional
- **External imports:** `rich.logging.RichHandler`
- **Hardcoded model names:** None
- **Training type:** N/A (infrastructure)

---

#### 2. `src/aiprod_trainer/captioning.py`
- **Lines:** 402
- **Summary:** Audio-visual media captioning system with two backends: (1) **Qwen2.5-Omni** — a local multimodal model for audio+video captioning, and (2) **Gemini Flash** — a Google API-based captioner. Provides `BaseCaptioner` ABC with `QwenOmniCaptioner` and `GeminiFlashCaptioner` implementations. Factory function `create_captioner()`. Handles video frame extraction, audio extraction via torchaudio, and caption cleanup (removing LLM preamble patterns).
- **Implementation:** ✅ Real, fully functional
- **External imports:** `transformers` (Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, BitsAndBytesConfig), `google.genai`, `torchaudio`, `av` (PyAV)
- **Hardcoded model names:**
  - `"Qwen/Qwen2.5-Omni-7B"` (line ~65)
  - `"gemini-2.0-flash-lite"` (line ~180, via `google.genai`)
- **Training type:** Preprocessing utility (not training)

---

#### 3. `src/aiprod_trainer/config.py`
- **Lines:** 521
- **Summary:** Comprehensive Pydantic v2 configuration system. Defines: `ModelConfig`, `LoraConfig`, `OptimizationConfig`, `AccelerationConfig`, `DataConfig`, `ValidationConfig`, `CheckpointsConfig`, `HubConfig`, `WandbConfig`, `FlowMatchingConfig`, and the root `AIPRODTrainerConfig`. Includes extensive validators (frame count % 8 == 1, dimensions divisible by 32, LoRA rank validation, learning rate bounds, etc.). Supports loading from YAML files.
- **Implementation:** ✅ Real, comprehensive with extensive validation logic
- **External imports:** `pydantic`, `yaml`
- **Hardcoded model names:** None (paths are configurable)
- **Training type:** Configuration infrastructure

---

#### 4. `src/aiprod_trainer/config_display.py`
- **Lines:** 137
- **Summary:** Rich-formatted console display of training configuration. Renders a multi-section table showing model, LoRA, optimization, data, validation, and acceleration settings.
- **Implementation:** ✅ Real, functional
- **External imports:** `rich` (Table, Console, Panel, Text)
- **Hardcoded model names:** None
- **Training type:** N/A (display utility)

---

#### 5. `src/aiprod_trainer/datasets.py`
- **Lines:** ~250
- **Summary:** Two dataset implementations: (1) `DummyDataset` — generates random tensors for benchmarking/smoke testing, and (2) `PrecomputedDataset` — loads precomputed `.pt` files containing VAE-encoded latents and text embeddings. Supports legacy patchified `[seq_len, C]` format and modern `[C, F, H, W]` format. Uses `einops` for format conversion.
- **Implementation:** ✅ Real, functional
- **External imports:** `torch`, `einops`
- **Hardcoded model names:** None
- **Training type:** Data loading for fine-tuning (operates on precomputed latents, not raw video)

---

#### 6. `src/aiprod_trainer/gemma_8bit.py`
- **Lines:** ~135
- **Summary:** Loads the Gemma text encoder in 8-bit precision using bitsandbytes quantization. Wraps `aiprod_core`'s `AVGemmaTextEncoderModel` with HuggingFace's `Gemma3ForConditionalGeneration` loaded in INT8. Handles weight loading from safetensors checkpoints.
- **Implementation:** ✅ Real, functional
- **External imports:** `transformers` (Gemma3ForConditionalGeneration, BitsAndBytesConfig), `bitsandbytes`, `aiprod_core` (SafetensorsModelStateDictLoader, AVGemmaTextEncoderModel, GemmaFeaturesExtractorProjLinear, AIPRODVGemmaTokenizer)
- **Hardcoded model names:**
  - References **Gemma 3** architecture via `Gemma3ForConditionalGeneration`
- **Training type:** Model loading utility (loads existing pretrained text encoder)

---

#### 7. `src/aiprod_trainer/gpu_utils.py`
- **Lines:** ~95
- **Summary:** GPU memory management utilities. `free_gpu_memory()` clears CUDA cache. `gpu_memory_manager` context manager / decorator for scoped memory cleanup. `get_gpu_memory_info()` queries nvidia-smi for per-GPU memory usage.
- **Implementation:** ✅ Real, functional
- **External imports:** `torch`, `subprocess` (for nvidia-smi)
- **Hardcoded model names:** None
- **Training type:** N/A (infrastructure)

---

#### 8. `src/aiprod_trainer/hf_hub_utils.py`
- **Lines:** ~225
- **Summary:** HuggingFace Hub integration. `push_to_hub()` uploads LoRA checkpoints with auto-generated model cards. Converts validation videos to GIFs for Hub display. Renders model card from Jinja-like template with training metadata.
- **Implementation:** ✅ Real, functional
- **External imports:** `huggingface_hub` (HfApi, create_repo, upload_folder), `imageio`, `safetensors`
- **Hardcoded model names:** None (base model name comes from config)
- **Training type:** Post-training utility

---

#### 9. `src/aiprod_trainer/model_loader.py`
- **Lines:** ~310
- **Summary:** Unified model loading using `aiprod_core`'s `SingleGPUModelBuilder`. Provides individual loader functions: `load_transformer()`, `load_video_vae_encoder()`, `load_video_vae_decoder()`, `load_audio_vae_encoder()`, `load_audio_vae_decoder()`, `load_vocoder()`, `load_text_encoder()`. All models are loaded from a single `.safetensors` checkpoint file.
- **Implementation:** ✅ Real, functional
- **External imports:** `aiprod_core` (SingleGPUModelBuilder, Transformer3DModel, CausalVideoAutoencoder variants, AudioVAEModel variants, BigVGAN16khzModel, AVGemmaTextEncoderModel, AIPRODVGemmaTokenizer)
- **Hardcoded model names:** None (checkpoint path is parameter)
- **Training type:** Model loading (loads existing pretrained model for fine-tuning)

---

#### 10. `src/aiprod_trainer/progress.py`
- **Lines:** 237
- **Summary:** Rich progress bars for training and validation. `TrainingProgress` shows epoch/step/loss/LR info. `StandaloneSamplingProgress` and `SamplingContext` handle validation sampling progress with per-step denoising display.
- **Implementation:** ✅ Real, functional
- **External imports:** `rich` (Progress, Live, Panel, Table, etc.)
- **Hardcoded model names:** None
- **Training type:** N/A (UI utility)

---

#### 11. `src/aiprod_trainer/quantization.py`
- **Lines:** ~195
- **Summary:** Post-training quantization using `optimum-quanto`. Quantizes model weights block-by-block to reduce VRAM. Supports int2/int4/int8/fp8 quantization types. Excludes specific module types from quantization (patchify_proj, adaln, norms, embeddings). Also provides optional weights-only quantization for inference.
- **Implementation:** ✅ Real, functional
- **External imports:** `optimum.quanto` (freeze, quantize, qint2/qint4/qint8/qfloat8)
- **Hardcoded model names:** None
- **Training type:** Optimization utility (reduces VRAM for fine-tuning)

---

#### 12. `src/aiprod_trainer/timestep_samplers.py`
- **Lines:** ~130
- **Summary:** Timestep sampling strategies for flow matching training. `UniformTimestepSampler` samples uniformly from [0,1]. `ShiftedLogitNormalTimestepSampler` samples from a logit-normal distribution with configurable shift based on image/video token count — biases toward higher noise for larger content.
- **Implementation:** ✅ Real, functional (includes mathematical formulation)
- **External imports:** `torch`
- **Hardcoded model names:** None
- **Training type:** Core training component (flow matching timestep sampling)

---

#### 13. `src/aiprod_trainer/trainer.py`
- **Lines:** 978
- **Summary:** **Main training orchestration class** `AIPRODvTrainer`. Implements the full training loop:
  1. Loads Gemma text encoder → caches validation prompt embeddings → unloads Gemma (memory optimization)
  2. Loads transformer + VAE via `model_loader`
  3. Optionally quantizes transformer via `quantization.py`
  4. Applies LoRA via PEFT (`peft.LoraConfig`, `get_peft_model`)
  5. Sets up AdamW/Adafactor/8-bit Adam optimizer
  6. Creates LR scheduler (linear, cosine, polynomial, constant warmup, etc.)
  7. Wraps everything with HuggingFace Accelerate (DDP/FSDP)
  8. Training loop: loads precomputed latents → applies training strategy → computes flow matching loss → backprop
  9. Periodic validation sampling, checkpoint saving, W&B logging, Hub pushing
- **Implementation:** ✅ Real, comprehensive, production-quality
- **External imports:** `accelerate`, `peft` (LoraConfig, get_peft_model, PeftModel), `torch`, `wandb`, `bitsandbytes` (bnb.optim.AdamW8bit)
- **Hardcoded model names:** None
- **Training type:** **Fine-tuning** — loads a pretrained AIPROD/LTX model and trains LoRA adapters (or full params) on custom data

---

#### 14. `src/aiprod_trainer/utils.py`
- **Lines:** ~100
- **Summary:** Image I/O utilities. `open_image_as_srgb()` loads images with ICC color profile conversion to sRGB. `save_image()` saves torch tensors as images.
- **Implementation:** ✅ Real, functional
- **External imports:** `PIL` (Image, ImageCms), `torchvision`
- **Hardcoded model names:** None
- **Training type:** N/A (utility)

---

#### 15. `src/aiprod_trainer/validation_sampler.py`
- **Lines:** 859
- **Summary:** `ValidationSampler` class for generating sample videos during training. Implements full denoising inference pipeline: noise generation → iterative denoising with Euler steps → VAE decoding → video/audio saving. Supports text-to-video, image-to-video, and video-to-video (IC-LoRA) modes. Uses Classifier-Free Guidance (CFG) and Spatio-Temporal Guidance (STG). Supports tiled VAE decoding for large videos.
- **Implementation:** ✅ Real, comprehensive
- **External imports:** `aiprod_core` (EulerDiffusionStep, CFGGuider, STGGuider, GaussianNoiser, RectifiedFlowScheduler, Patchifier variants, X0Model, etc.)
- **Hardcoded model names:** None
- **Training type:** Inference/validation within training pipeline

---

#### 16. `src/aiprod_trainer/video_utils.py`
- **Lines:** ~170
- **Summary:** Video I/O using PyAV. `read_video()` decodes video frames with optional frame limit. `save_video()` encodes video with H.264 and optional AAC audio. `get_video_frame_count()` fast frame counting.
- **Implementation:** ✅ Real, functional
- **External imports:** `av` (PyAV), `torch`, `torchaudio`
- **Hardcoded model names:** None
- **Training type:** N/A (utility)

---

### B. Streaming Module — `src/aiprod_trainer/streaming/`

---

#### 17. `src/aiprod_trainer/streaming/__init__.py`
- **Lines:** 37
- **Summary:** Public API exports: `StreamingDatasetAdapter`, `SmartLRUCache`, `AsyncPrefetcher`, `DataSourceConfig`.
- **Implementation:** ✅ Real
- **External imports:** Local submodules only
- **Hardcoded model names:** None
- **Training type:** N/A

---

#### 18. `src/aiprod_trainer/streaming/adapter.py`
- **Lines:** 320
- **Summary:** `StreamingDatasetAdapter` — a PyTorch `Dataset` that supports multi-source data loading (local, HuggingFace Hub, S3, GCS) with LRU caching and async prefetching. Maps source names to output keys via `data_mapping`. Not yet integrated into the main training loop (alternative to `PrecomputedDataset`).
- **Implementation:** ✅ Real, functional
- **External imports:** `torch.utils.data.Dataset`
- **Hardcoded model names:** None
- **Training type:** Data loading infrastructure

---

#### 19. `src/aiprod_trainer/streaming/cache.py`
- **Lines:** ~330
- **Summary:** `SmartLRUCache` with zstd compression, TTL expiration, and detailed metrics (hits, misses, evictions, compression savings). `AsyncPrefetcher` runs a background async worker that pre-fetches upcoming data items into the cache. Thread-safe via `asyncio.Lock`.
- **Implementation:** ✅ Real, functional
- **External imports:** `zstandard` (zstd compression)
- **Hardcoded model names:** None
- **Training type:** Data pipeline infrastructure

---

#### 20. `src/aiprod_trainer/streaming/sources.py`
- **Lines:** ~300
- **Summary:** `DataSource` abstract base class with four implementations: `LocalDataSource` (filesystem), `HuggingFaceDataSource` (HF Hub), `S3DataSource` (AWS), `GCSDataSource` (Google Cloud Storage). Each supports async file fetching, listing, and prefetching. Uses `DataSourceConfig` Pydantic model.
- **Implementation:** ✅ Real, functional
- **External imports:** `boto3` (S3), `google.cloud.storage` (GCS), `huggingface_hub`
- **Hardcoded model names:** None
- **Training type:** Data pipeline infrastructure

---

### C. Training Strategies — `src/aiprod_trainer/training_strategies/`

---

#### 21. `src/aiprod_trainer/training_strategies/__init__.py`
- **Lines:** ~55
- **Summary:** Factory function `get_training_strategy()` — returns `TextToVideoStrategy` or `VideoToVideoStrategy` based on config type string using `match/case`.
- **Implementation:** ✅ Real
- **External imports:** Local submodules
- **Hardcoded model names:** None
- **Training type:** Strategy routing

---

#### 22. `src/aiprod_trainer/training_strategies/base_strategy.py`
- **Lines:** ~260
- **Summary:** Abstract `TrainingStrategy` base class. Defines `ModelInputs` dataclass (latents, embeddings, masks, timesteps, position IDs). Provides shared logic: position embedding generation via `aiprod_core` patchifiers, conditioning mask creation, per-token timestep broadcasting.
- **Implementation:** ✅ Real, functional
- **External imports:** `aiprod_core` (SymmetricPatchifier / AVPatchifier)
- **Hardcoded model names:** None
- **Training type:** Core training abstraction

---

#### 23. `src/aiprod_trainer/training_strategies/text_to_video.py`
- **Lines:** ~285
- **Summary:** `TextToVideoStrategy` for text-to-video LoRA training. Implements flow matching: samples noise, computes noisy latents via `noise * t + clean * (1-t)`, velocity target is `noise - clean`. Supports optional audio branch with separate loss weighting. First-frame conditioning for image-to-video.
- **Implementation:** ✅ Real, functional
- **External imports:** `torch`, `aiprod_core`
- **Hardcoded model names:** None
- **Training type:** **Fine-tuning** — flow matching loss on precomputed latents

---

#### 24. `src/aiprod_trainer/training_strategies/video_to_video.py`
- **Lines:** ~290
- **Summary:** `VideoToVideoStrategy` for IC-LoRA (Image Conditioning LoRA) training. Concatenates reference video latents with target video latents along channel dim. Adds conditioning masks. Loss computed only on target portion (not reference). Infers reference downscale factor from latent dimensions.
- **Implementation:** ✅ Real, functional
- **External imports:** `torch`, `aiprod_core`
- **Hardcoded model names:** None
- **Training type:** **Fine-tuning** — IC-LoRA for video-to-video transformation

---

### D. Scripts — `scripts/`

---

#### 25. `scripts/train.py`
- **Lines:** ~65
- **Summary:** CLI entry point using Typer. Loads YAML config, validates, creates `AIPRODvTrainer`, calls `train()`. Supports optional `--override` for key=value config overrides.
- **Implementation:** ✅ Real, functional
- **External imports:** `typer`, `yaml`
- **Hardcoded model names:** None
- **Training type:** Entry point for fine-tuning

---

#### 26. `scripts/inference.py`
- **Lines:** 444
- **Summary:** Full inference CLI. Supports text-to-video (T2V), image-to-video (I2V), and video-to-video (V2V) modes. Auto-detects LoRA rank and target modules from checkpoint metadata. Loads base model via `aiprod_core` pipeline, applies LoRA adapter via PEFT, generates video/audio with configurable guidance/steps/seed.
- **Implementation:** ✅ Real, comprehensive
- **External imports:** `peft` (PeftModel, LoraConfig), `aiprod_core` (full pipeline), `safetensors`
- **Hardcoded model names:** None (all paths from CLI args)
- **Training type:** Inference (uses fine-tuned LoRA)

---

#### 27. `scripts/process_dataset.py`
- **Lines:** 318
- **Summary:** Orchestration script that runs the full preprocessing pipeline: (1) compute caption embeddings via `process_captions.py`, (2) compute video latents via `process_videos.py`, and optionally (3) compute reference videos for IC-LoRA via `compute_reference.py`. Calls subprocess for each step.
- **Implementation:** ✅ Real, functional
- **External imports:** `subprocess`, `typer`
- **Hardcoded model names:** None
- **Training type:** Preprocessing orchestrator

---

#### 28. `scripts/process_videos.py`
- **Lines:** 1040
- **Summary:** Largest script. `MediaDataset` class handles video/image loading, resolution bucketing (aspect-ratio-aware nearest bucket selection), resize-and-crop (center or random), audio extraction via torchaudio. `compute_latents()` encodes media through the VAE encoder and saves latent `.pt` files. `encode_video()` handles standard and tiled encoding. `encode_audio()` converts waveforms to mel spectrograms via `AudioProcessor` then through audio VAE. `tiled_encode_video()` implements spatial tiling with feathered blending for large resolution support.
- **Implementation:** ✅ Real, comprehensive, production-quality
- **External imports:** `torch`, `torchaudio`, `torchvision`, `av`, `aiprod_core` (AudioProcessor), `numpy`, `einops`
- **Hardcoded model names:** None
- **Training type:** Preprocessing (VAE encoding of training data)

---

#### 29. `scripts/process_captions.py`
- **Lines:** 429
- **Summary:** `CaptionsDataset` loads captions from CSV/JSON/JSONL metadata files. Cleans LLM-generated prefixes (e.g., "Here is a caption:" patterns). Computes text embeddings through Gemma text encoder via `_preprocess_text()` and saves as `.pt` files. Supports optional LoRA trigger word prepending and 8-bit text encoder loading.
- **Implementation:** ✅ Real, functional
- **External imports:** `torch`, `aiprod_core` (text encoder), `aiprod_trainer` (model_loader, gemma_8bit)
- **Hardcoded model names:** None (encoder path from CLI)
- **Training type:** Preprocessing (text embedding computation)

---

#### 30. `scripts/caption_videos.py`
- **Lines:** 487
- **Summary:** Auto-captioning CLI. Scans directories for media files, captions them using Qwen2.5-Omni or Gemini Flash (via `captioning.py`), saves results in TXT/CSV/JSON/JSONL format. Supports incremental captioning (skip existing), recursive directory scanning, configurable FPS sampling, audio inclusion toggle.
- **Implementation:** ✅ Real, functional
- **External imports:** `typer`, `torch`, `aiprod_trainer.captioning`
- **Hardcoded model names:** Indirect via `captioning.py` (Qwen2.5-Omni-7B, Gemini Flash)
- **Training type:** Preprocessing utility

---

#### 31. `scripts/decode_latents.py`
- **Lines:** 370
- **Summary:** `LatentsDecoder` class decodes precomputed `.pt` latent files back into videos/images using the VAE decoder. Supports tiled decoding for large resolutions. Handles both old patchified and new non-patchified latent formats. Also decodes audio latents via audio VAE + vocoder.
- **Implementation:** ✅ Real, functional
- **External imports:** `torch`, `torchaudio`, `torchvision`, `einops`, `aiprod_core` (VAE tiling configs)
- **Hardcoded model names:** None
- **Training type:** Verification/debugging utility

---

#### 32. `scripts/compute_reference.py`
- **Lines:** 289
- **Summary:** Computes reference videos for IC-LoRA training using Canny edge detection. Processes each video frame through OpenCV's Canny detector, saves 3-channel edge maps as reference videos. Updates captions JSON with reference_path entries.
- **Implementation:** ✅ Real, functional
- **External imports:** `cv2` (OpenCV), `torch`, `torchvision`, `typer`
- **Hardcoded model names:** None
- **Training type:** Preprocessing for IC-LoRA

---

#### 33. `scripts/split_scenes.py`
- **Lines:** 418
- **Summary:** Video scene splitting using PySceneDetect. Supports Content, Adaptive, Threshold, and Histogram detectors. Splits videos via ffmpeg. Generates preview images and HTML scene reports. Configurable minimum scene length, duration filtering, frame skipping, downscaling.
- **Implementation:** ✅ Real, functional
- **External imports:** `scenedetect` (ContentDetector, AdaptiveDetector, etc.), `ffmpeg` (via scenedetect splitter)
- **Hardcoded model names:** None
- **Training type:** Dataset preparation utility

---

### E. Configuration Files

---

#### 34. `configs/ltx2_av_lora.yaml`
- **Lines:** 314
- **Summary:** Standard audio-video LoRA training configuration. LoRA rank 32, target modules include attention projections + feedforward. Learning rate 3e-5, AdamW optimizer, cosine schedule with 100 warmup steps. Resolution 768×512×49 frames. Placeholder model paths.
- **Implementation:** ✅ Real, well-documented
- **Hardcoded model names:** `"path/to/AIPROD-2-model.safetensors"` (placeholder)

---

#### 35. `configs/ltx2_av_lora_low_vram.yaml`
- **Lines:** 326
- **Summary:** Low-VRAM variant for 24-32GB GPUs. INT8 quantization enabled, 8-bit Adam optimizer, LoRA rank 16, gradient checkpointing, smaller resolution (512×384×33).
- **Implementation:** ✅ Real, well-documented
- **Hardcoded model names:** `"path/to/AIPROD-2-model.safetensors"` (placeholder)

---

#### 36. `configs/ltx2_v2v_ic_lora.yaml`
- **Lines:** 330
- **Summary:** IC-LoRA video-to-video config. Training strategy set to `"video_to_video"`. LoRA targets include explicit video module projections. Reference downscale factor 2. Validation uses reference video inputs.
- **Implementation:** ✅ Real, well-documented
- **Hardcoded model names:** `"path/to/AIPROD-2-model.safetensors"` (placeholder)

---

#### 37–40. `configs/accelerate/ddp.yaml`, `ddp_compile.yaml`, `fsdp.yaml`, `fsdp_compile.yaml`
- **Lines:** 17–32 each
- **Summary:** HuggingFace Accelerate distributed training configs. DDP (multi-GPU data parallel) and FSDP (fully sharded data parallel) variants, each with an optional `torch.compile` (Inductor backend) variant. All use bf16 mixed precision, 4 GPU processes.
- **Implementation:** ✅ Real configs
- **Hardcoded model names:** `BasicAVTransformerBlock` (FSDP wrap policy — references internal model architecture from aiprod-core)

---

### F. Tests — `tests/streaming/`

---

#### 41. `tests/streaming/conftest.py`
- **Lines:** 82
- **Summary:** Shared pytest fixtures: `temp_data_dir` (creates 100 synthetic latents + conditions), `sample_tensor_dict`, `async_event_loop`, `mock_fetch_fn`.
- **Implementation:** ✅ Real test fixtures

#### `tests/streaming/test_adapter.py`
- **Lines:** 391
- **Summary:** Integration tests for `StreamingDatasetAdapter` — creation, multi-source, prefetch, DataLoader compatibility, cache clearing.
- **Implementation:** ✅ Real tests (8+ test functions)

#### `tests/streaming/test_cache.py`
- **Lines:** 325
- **Summary:** Unit tests for `SmartLRUCache` — put/get, hit rate tracking, LRU eviction, TTL expiration, zstd compression, concurrent access, cache clearing.
- **Implementation:** ✅ Real tests (10+ test functions)

#### `tests/streaming/test_prefetcher.py`
- **Lines:** 332
- **Summary:** Unit tests for `AsyncPrefetcher` — start/stop lifecycle, queue behavior, already-cached skip, hit tracking, concurrent prefetch.
- **Implementation:** ✅ Real tests (8+ test functions)

#### `tests/streaming/test_sources.py`
- **Lines:** 255
- **Summary:** Unit tests for `LocalDataSource` — uncompressed/compressed fetch, file listing, concurrent fetch, prefetch, error handling, compression ratio benchmarks.
- **Implementation:** ✅ Real tests (8+ test functions)

#### `tests/streaming/test_performance.py`
- **Lines:** 372
- **Summary:** Performance benchmarks — local fetch speed, cache hit/miss speed, compression encode/decode speed, throughput with/without cache, throughput with prefetch.
- **Implementation:** ✅ Real benchmarks using pytest-benchmark

#### `tests/streaming/run_tests.py`
- **Lines:** 63
- **Summary:** Test runner script with options for unit/integration/benchmark test filtering and coverage reporting.
- **Implementation:** ✅ Real

#### `tests/streaming/__init__.py`
- **Lines:** 12
- **Summary:** Module docstring describing test organization.
- **Implementation:** ✅ Real

---

### G. Other Files

---

#### 42. `templates/model_card.md`
- **Lines:** 52
- **Summary:** Jinja-style template for HuggingFace model cards. Includes placeholders for model name, base model, training type, steps, learning rate, sample grid, and validation prompts. References Lightricks and AIPROD GitHub.
- **Implementation:** ✅ Real template
- **Hardcoded references:** `"Lightricks/AIPROD"` (base model HF link), `"https://github.com/Lightricks/AIPROD"` (trainer repo)

#### `pyproject.toml`
- **Lines:** ~98
- **Summary:** Package metadata. Name: `aiprod-trainer`. Python ≥3.10. Key dependencies: `aiprod-core`, `accelerate>=1.2.0`, `peft>=0.14.0`, `torch>=2.5.0`, `wandb`, `optimum-quanto`, `bitsandbytes`, `huggingface-hub`, `av`, `torchaudio`, `torchvision`, `pydantic>=2.0`, `rich`, `typer`, `einops`, `safetensors`, `Pillow`, `imageio`, `PyYAML`.
- **Implementation:** ✅ Real

#### `pytest.ini`
- **Lines:** 42
- **Summary:** Test configuration with async support, custom markers (slow, benchmark, integration, unit), test discovery paths.
- **Implementation:** ✅ Real

#### `AGENTS.md`
- **Lines:** 353
- **Summary:** Developer guide for AI coding assistants. Documents architecture overview, key patterns, development commands, code standards.
- **Implementation:** ✅ Real documentation

#### `CLAUDE.md`
- **Lines:** 1
- **Summary:** Points to AGENTS.md.

#### `README.md`
- **Lines:** 53
- **Summary:** Package overview. States "PROPRIETARY - STRICTLY CONFIDENTIAL © 2026 Blockprod". Describes the package as training/fine-tuning tools for AIPROD model. Links to documentation. Recommends 80GB+ VRAM (or 32GB with low-VRAM config).
- **Implementation:** ✅ Real documentation

#### `docs/2026-01-29/` (8 markdown files)
- **Summary:** Detailed documentation: training modes, training guide, utility scripts, troubleshooting, custom training strategies, configuration reference, quick start, dataset preparation.
- **Implementation:** Documentation only (not audited line-by-line)

---

## Summary Tables

### External Model Dependencies

| Model | Where Referenced | Purpose |
|-------|-----------------|---------|
| **Qwen/Qwen2.5-Omni-7B** | `captioning.py` line ~65 | Auto-captioning (local multimodal model) |
| **gemini-2.0-flash-lite** | `captioning.py` line ~180 | Auto-captioning (Google API) |
| **Gemma 3** | `gemma_8bit.py`, `model_loader.py` | Text encoder for AIPROD |
| **AIPROD/LTX-Video** (transformer + VAE) | `model_loader.py`, `trainer.py`, `validation_sampler.py` | Base model being fine-tuned |
| **BigVGAN-16kHz** | `model_loader.py` | Audio vocoder |

### External Library Dependencies

| Library | Purpose |
|---------|---------|
| `aiprod-core` | All model architectures (transformer, VAE, text encoder, scheduler, patchifier) |
| `peft` | LoRA adapter creation and training |
| `accelerate` | Distributed training (DDP, FSDP) |
| `optimum-quanto` | Weight quantization (int8/int4/fp8) |
| `bitsandbytes` | 8-bit optimizer, 8-bit model loading |
| `transformers` | Gemma3, Qwen2.5-Omni loading |
| `wandb` | Experiment tracking |
| `huggingface-hub` | Model upload |
| `scenedetect` | Video scene splitting |
| `av` (PyAV) | Video I/O |
| `torchaudio` | Audio I/O |
| `zstandard` | Data compression for streaming cache |
| `google.genai` | Gemini Flash API |
| `boto3` | AWS S3 access |
| `google.cloud.storage` | GCS access |
| `cv2` (OpenCV) | Canny edge detection |

### Implementation Status

| Category | Files | Status |
|----------|-------|--------|
| Core library modules | 16 | ✅ 100% real implementation |
| Training strategies | 3 | ✅ 100% real implementation |
| Streaming module | 4 | ✅ 100% real implementation |
| Scripts | 8 | ✅ 100% real implementation |
| Tests | 8 | ✅ 100% real tests |
| Configs | 7 | ✅ 100% real configs |
| **Total** | **46** | **✅ Zero stubs/placeholders** |

---

## Final Verdict

### Is this training a proprietary model from scratch, or fine-tuning an existing open-source model?

**This is fine-tuning an existing model.** Specifically:

1. The base model is **AIPROD** (branded as Lightricks LTX-Video), an open-source audio-video generation model based on a transformer diffusion architecture with flow matching.

2. All model architectures come from the `aiprod-core` package — the trainer creates **no new model architectures**.

3. Training operates on **precomputed latents** — raw videos are VAE-encoded offline, and the trainer only learns LoRA adapter weights (or optionally fine-tunes full transformer weights) in latent space.

4. The text encoder (Gemma) is **frozen** during training — it's loaded only to precompute text embeddings, then immediately unloaded from GPU.

5. The VAE encoder/decoder are **frozen** — only the transformer backbone receives gradient updates.

6. The default and recommended training mode is **LoRA** (Low-Rank Adaptation), which adds small trainable matrices to attention/feedforward layers while keeping the base model frozen.

7. The model card template explicitly states: *"This is a fine-tuned version of [base_model]"*.
