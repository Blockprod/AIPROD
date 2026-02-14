# PHASE 2 LAUNCH — Pipeline Cinématographique Complet (Weeks 13-24)

**Date:** February 14, 2026  
**Status:** 🚀 LAUNCHED  
**Duration:** 12 weeks (Weeks 13-24)  
**Objective:** Implement all remaining cinematic components for end-to-end video generation

---

## Phase 2 Overview

Phase 1 established the **foundational infrastructure** (core API, trainers, test suite, data governance). Phase 2 now builds the **complete cinematic pipeline** with all missing components:

1. **TTS Module** — Text-to-speech for voice synthesis
2. **Lip-Sync Module** — Audio-to-facial animation sync
3. **Audio Mixer** — Multi-track audio processing (voice, music, FX, ambient)
4. **Montage Engine** — Automatic scene editing and timeline generation
5. **Color Grading** — Professional color science and LUT grading
6. **Export Engine** — Multi-format video output (H.265, ProRes, AV1, etc.)
7. **Inference Node Connection** — Real computation replacing mocked `torch.randn()`

**Target Completion:** May 30, 2026 (12 weeks from today)

---

## Phase 2 Work Breakdown

### Step 2.1: TTS Module [Weeks 1-3]

**Files Created:**
```
packages/aiprod-core/src/aiprod_core/model/tts/
├── __init__.py
├── model.py              # TTSModel + TTSConfig
├── text_frontend.py      # TODO: G2P, text normalization
├── prosody.py            # TODO: F0 prediction, duration modeling
├── speaker_embedding.py  # TODO: Multi-speaker embeddings
└── vocoder_tts.py        # TODO: Specialized vocoder
```

**Key Components to Implement:**
1. **Text Frontend**
   - G2P (Grapheme-to-Phoneme) conversion
   - Text normalization (numbers, abbreviations, special chars)
   - Phoneme tokenization

2. **Prosody Modeling**
   - F0 (fundamental frequency) prediction
   - Duration prediction per phoneme
   - Energy modeling

3. **TTS Encoder-Decoder**
   - Text encoder: Position enc + transformer blocks
   - Decoder: Predictive decoder for mel-spectrogram
   - Multi-speaker support via speaker embeddings

4. **Vocoder**
   - Mel-spectrogram → Waveform conversion
   - High-quality speech reconstruction
   - Real-time capable

**Training Data:**
- LibriTTS: 585 hours, multi-speaker
- Common Voice: Multi-language, 400+ hours
- Optional: Commercial voice datasets

**Quality Target:** MOS ≥ 4.0 (comparable to ElevenLabs)

**Success Criteria:**
- ✅ TTS model generates intelligible speech
- ✅ Multi-language support (EN, FR, ES, DE, IT, JA, ZH)
- ✅ Multi-speaker with speaker embedding
- ✅ Real-time inference (< 100ms per second of audio)
- ✅ MOS score ≥ 4.0 on test set

---

### Step 2.2: Lip-Sync Module [Weeks 4-6]

**Files Created:**
```
packages/aiprod-core/src/aiprod_core/model/lip_sync/
├── __init__.py
├── model.py              # LipSyncModel + LipSyncConfig
├── sync_loss.py          # TODO: LSE-D and LSE-C metrics
├── FLAME_mapper.py       # TODO: FLAME param mapping
└── inference.py          # TODO: Real-time inference pipeline
```

**Components:**
1. **Audio Encoder**
   - Mel-spectrogram feature extraction
   - Temporal context (±5 frames)

2. **Facial Parameter Decoder**
   - FLAME blend shape prediction (52 facial params)
   - Bi-directional LSTM or Transformer
   - Attention mechanism for audio-visual alignment

3. **Loss Functions**
   - LSE-D (Lip-Sync Error Distance): Euclidean distance in parameter space
   - LSE-C (Lip-Sync Error Confidence): Correlation with voiced regions
   - Sync loss should be < 7.0 for good quality

4. **Inference Pipeline**
   - Real-time processing (30 fps)
   - Sliding window inference
   - Smooth temporal predictions

**Training Data:**
- LRS2 2.0: 145 hours, in-the-wild videos
- VoxCeleb2: 1M+ videos, facial data available
- Custom scenarios (various facial shapes, lighting)

**Success Criteria:**
- ✅ LSE-D ≤ 7.0 (tight audio-visual sync)
- ✅ LSE-C ≥ 6.0 (high confidence in voiced regions)
- ✅ Real-time inference (30 fps on GPU)
- ✅ Works across facial types (age, ethnicity, gender)
- ✅ Smooth temporal predictions (no jitter)

---

### Step 2.3: Audio Mixer [Weeks 7-8]

**Files Created:**
```
packages/aiprod-core/src/aiprod_core/model/audio_mixer/
├── __init__.py
├── mixer.py              # AudioMixer + AudioTrack classes
├── spatial_audio.py      # TODO: Stereo, 5.1, binaural
├── dynamics.py           # TODO: Compressor, EQ, limiter
└── music_controller.py   # TODO: Conditional music generation
```

**Components:**
1. **Multi-Track Mixing**
   - Voice track (from TTS)
   - Music track
   - Ambient/foley effects
   - Voiceover (optional)
   - Per-track volume, pan, mute, solo controls

2. **Dynamic Effects**
   - Compressor (attack, release, ratio, threshold)
   - Parametric EQ (graphic or parametric)
   - Reverb (convolution or algorithmic)
   - Hard/soft limiter for loudness control

3. **Spatial Audio**
   - Stereo mixing (pan law: -3dB center)
   - 5.1 surround creation
   - Binaural HRTF processing

4. **Audio Normalization**
   - LUFS (Loudness Units relative to Full Scale) normalization
   - Target: -14 LUFS (YouTube standard)
   - Preventing distortion via limiting

**Success Criteria:**
- ✅ Seamless multi-track mixing
- ✅ Proper gain staging (no clipping)
- ✅ Perceptually good compression (natural sound)
- ✅ LUFS ≈ -14 (YouTube compliance)
- ✅ Binaural audio tested with headphones

---

### Step 2.4: Montage Engine [Weeks 9-10]

**Files Created:**
```
packages/aiprod-pipelines/src/aiprod_pipelines/editing/
├── __init__.py
├── timeline.py           # TimelineGenerator, Clips, Transitions
├── transitions.py        # TODO: Cross-fade, wipe, match-cut transitions
├── pacing_engine.py      # TODO: Rhythm control, shot duration optimization
├── continuity_checker.py # TODO: Scene continuity verification
└── timeline_export.py    # TODO: EDL, FCPXML, AAF export
```

**Components:**
1. **Timeline Generation**
   - Scene-to-scene sequencing
   - Automatic duration calculation
   - Transition placement and type selection

2. **Transition Effects**
   - Instant cut (no transition)
   - Cross-fade (opacity blend)
   - Dissolve (cross-fade with effect)
   - Wipe (directional transition)
   - Match-cut (spatial/content continuity)

3. **Pacing Engine**
   - Emotion-aware timing (action → short shots, drama → long shots)
   - Complexity-based duration (visual complexity → longer for comprehension)
   - Narrative rhythm control (building tension → accelerate cuts)

4. **Continuity Checking**
   - Color continuity (flagging mismatches)
   - Spatial continuity (camera position, actor positions)
   - Temporal continuity (no time jumps)

5. **Standard Export Formats**
   - EDL (Edit Decision List): Standard interchange format
   - FCPXML: Final Cut Pro XML for import/export
   - AAF: Advanced Authoring Format for Avid systems

**Success Criteria:**
- ✅ Seamless scene transitions
- ✅ Pacing matches content emotion/action level
- ✅ Exported timelines importable in professional NLEs (Premiere, Final Cut, Avid)
- ✅ No continuity violations (or flagged explicitly)
- ✅ Multi-format export tested

---

### Step 2.5: Color Grading [Weeks 11-12]

**Files Created:**
```
packages/aiprod-pipelines/src/aiprod_pipelines/color/
├── __init__.py
├── color_pipeline.py     # Main pipeline
├── lut_manager.py        # TODO: 3D LUT application
├── color_space.py        # TODO: Rec.709, Rec.2020, DCI-P3, ACES conversion
├── hdr.py                # TODO: Tone mapping, HDR10, Dolby Vision
├── auto_grade.py         # TODO: AI-based color grading
├── scene_matching.py     # TODO: Cross-scene color consistency
└── luts/                 # LUT library (to populate)
    ├── cinematic_warm.cube
    ├── cinematic_cold.cube
    ├── documentary.cube
    ├── corporate.cube
    └── ... (20+ LUTs)
```

**Components:**
1. **3D LUT Application**
   - Load .cube files (industry standard)
   - Trilinear interpolation for color lookup
   - Blend multiple LUTs
   - Real-time application

2. **Color Space Conversions**
   - Rec.709 (SDR, broadcast)
   - Rec.2020 (HDR, wider gamut)
   - DCI-P3 (cinema)
   - ACES (standard for VFX)
   - Linear ↔ Log space conversions

3. **HDR Pipeline**
   - Tone mapping SDR → HDR (HABLE, filmic, ACES)
   - HDR10 metadata generation (SMPTE 2086)
   - Dolby Vision (if applicable)
   - Peak brightness control (1000 nits typical)

4. **Auto Color Grading (AI)**
   - Neural network trained on professional grade movies
   - Predicts optimal color adjustments per frame
   - Style-aware (cinematic, documentary, corporate, etc.)

5. **Scene Color Matching**
   - Compute color statistics per scene
   - Histogram matching between adjacent scenes
   - Automatic correction for lighting continuity

**Built-in LUT Library (to Create):**
- Cinematic (warm, cold, noir, high-contrast variants)
- Documentary (natural, neutral)
- Corporate (clean, professional)
- Vintage (film stock emulations)
- Sci-Fi (futuristic, high-saturation)
- Plus 10+ more

**Success Criteria:**
- ✅ Professional LUT application
- ✅ HDR workflow tested (export + playback)
- ✅ Color matching verified across scenes
- ✅ AI grading produces cinema-worthy results
- ✅ Multi-color-space support confirmed

---

### Step 2.6: Export Multi-Format [Weeks 13-14]

**Files Created:**
```
packages/aiprod-pipelines/src/aiprod_pipelines/export/
├── __init__.py
├── multi_format.py       # ExportEngine + profiles
├── video_encoder.py      # TODO: H.264, H.265, ProRes, etc.
├── audio_encoder.py      # TODO: AAC, OPUS, FLAC
├── muxer.py              # TODO: MP4, MXF, WebM muxing
└── profiles.py           # TODO: Preset configurations
```

**Supported Formats:**

| Format | Use Case | Priority | Codec |
|--------|----------|----------|-------|
| H.264 + AAC (.mp4) | Web, social | ✅ P0 | Existing |
| H.265 + AAC (.mp4) | Streaming HD/4K | P1 | New |
| ProRes 422 (.mov) | Professional editing | P1 | New |
| ProRes 4444 (.mov) | VFX compositing (alpha) | P1 | New |
| DNxHR (.mxf) | Avid broadcast | P2 | New |
| VP9 + OPUS (.webm) | Web (Google/Mozilla) | P2 | New |
| AV1 + OPUS (.webm) | Next-gen web | P2 | New |
| EXR sequence | VFX pipeline | P3 | New |
| DPX sequence | Digital cinema (DCP) | P3 | New |

**Export Profiles:**
- `web_mp4`: 1080p H.264, 8 Mbps (optimized for web)
- `streaming_hq`: 4K H.265, 25 Mbps (streaming platforms)
- `prores_editing`: ProRes 422, high bitrate (post-production)
- `dnxhr_mxf`: DNxHR, broadcast-quality
- `web_av1`: AV1, ultra-efficient compression

**Success Criteria:**
- ✅ All P1 formats working (H.265, ProRes, DNxHR)
- ✅ Metadata correct (color space tags, HDR flags)
- ✅ Audio sync verified across formats
- ✅ File sizes as expected (compression ratios match specs)
- ✅ Exported files playable in target NLEs/players

---

### Step 2.7: Connect Inference Nodes [Weeks 15-16]

**Current Status:** All 5 inference nodes return `torch.randn()` (mocked)

**Nodes to Connect:**

| Node | Current | Real Implementation |
|------|---------|-------------------|
| `TextEncodeNode` | Mocked | `AVGemmaTextEncoderModel` (from Step 1.1) |
| `DenoiseNode` | `torch.randn()` | `euler_denoising_loop()` or `denoise_audio_video()` |
| `UpsampleNode` | `torch.randn()` | `LatentUpsampler` (from Step 1) |
| `DecodeVideoNode` | `torch.randn()` | `VideoDecoder` (tiled VAE) |
| `AudioDecodeNode` | `torch.randn()` | `AudioVAEDecoder` + `Vocoder` |

**New Nodes to Add (from Phase 2):**
- `TTSNode` → TTS model from Step 2.1
- `LipSyncNode` → Lip-sync model from Step 2.2
- `AudioMixerNode` → Multi-track mixer from Step 2.3
- `TimelineNode` → Montage engine from Step 2.4
- `ColorGradeNode` → Color pipeline from Step 2.5
- `ExportNode` → Multi-format export from Step 2.6

**Impact:** Once connected, the ~62K lines of inference pipeline become **fully functional** (no longer mock computations).

**Success Criteria:**
- ✅ All nodes call real computation functions
- ✅ End-to-end pipeline produces actual video (not random tensors)
- ✅ Inference graph connects: Prompt → Video → TTS → LipSync → Export
- ✅ Performance benchmarked (inference time, memory usage)
- ✅ Error handling for failed nodes

---

## Module Directory Structure

```
Phase 2 deliverables:

aiprod-core/src/aiprod_core/model/
├── tts/                          [Step 2.1]
│   ├── model.py
│   ├── text_frontend.py
│   ├── prosody.py
│   ├── speaker_embedding.py
│   └── vocoder_tts.py
├── lip_sync/                     [Step 2.2]
│   ├── model.py
│   ├── sync_loss.py
│   ├── FLAME_mapper.py
│   └── inference.py
└── audio_mixer/                  [Step 2.3]
    ├── mixer.py
    ├── spatial_audio.py
    ├── dynamics.py
    └── music_controller.py

aiprod-pipelines/src/aiprod_pipelines/
├── editing/                      [Step 2.4]
│   ├── timeline.py
│   ├── transitions.py
│   ├── pacing_engine.py
│   ├── continuity_checker.py
│   └── timeline_export.py
├── color/                        [Step 2.5]
│   ├── color_pipeline.py
│   ├── lut_manager.py
│   ├── color_space.py
│   ├── hdr.py
│   ├── auto_grade.py
│   ├── scene_matching.py
│   └── luts/                     [20+ LUT files]
├── export/                       [Step 2.6]
│   ├── multi_format.py
│   ├── video_encoder.py
│   ├── audio_encoder.py
│   ├── muxer.py
│   └── profiles.py
└── inference/                    [Step 2.7]
    └── nodes_updated.py          [Replace mock torch.randn()]
```

---

## Implementation Strategy

### Recommended Order:
1. **Step 2.1: TTS** (3 weeks) — Foundation for voice
2. **Step 2.2: Lip-Sync** (2 weeks) — Depends on TTS output
3. **Step 2.3: Audio Mixer** (2 weeks) — Orthogonal (can parallel)
4. **Step 2.4: Montage** (2 weeks) — Orthogonal to TTS/audio
5. **Step 2.5: Color Grading** (2 weeks) — Orthogonal
6. **Step 2.6: Export** (1 week) — Depends on all above
7. **Step 2.7: Connect Nodes** (1 week) — Final integration

### Parallelization Opportunities:
- **Weeks 1-3:** Step 2.1 (TTS)
- **Weeks 1-3:** Step 2.4 (Montage) — parallel
- **Weeks 4-6:** Step 2.2 (Lip-Sync)
- **Weeks 7-8:** Step 2.3 (Audio Mixer) — parallel with 2.2
- **Weeks 9-10:** Step 2.5 (Color Grading) — parallel
- **Weeks 11-14:** Step 2.6 (Export)
- **Week 15-16:** Step 2.7 (Node Connection + Integration)

---

## Testing Strategy

### Unit Tests (per step):
```python
# For TTS:
test_tts_text_normalization()
test_tts_prosody_prediction()
test_tts_inference_quality()
test_tts_multilingual()

# For Lip-Sync:
test_lipsync_audio_processing()
test_lipsync_param_prediction()
test_lipsync_metrics(lse_d, lse_c)

# For Mixer:
test_mixer_multi_track()
test_mixer_compression()
test_mixer_loudness_normalization()

# For Montage:
test_timeline_generation()
test_transition_quality()
test_pacing_algorithm()

# For Color:
test_lut_application()
test_color_space_conversion()
test_hdr_tone_mapping()
test_scene_color_matching()

# For Export:
test_export_all_formats()
test_video_audio_sync()
test_metadata_tags()

# For Node Connection:
test_end_to_end_inference()
test_node_pipeline_real_computation()
```

### Integration Tests:
```python
test_full_pipeline_prompt_to_export()
test_multilingual_tts_with_lipsync()
test_color_graded_export_formats()
test_performance_benchmarks()
```

---

## Dependencies & Tools

### TTS:
- `librosa` — Audio processing
- `g2p-en`, `g2p-fr` — Grapheme-to-phoneme
- `WaveGlow` or `HiFi-GAN` — Vocoder (pre-trained)

### Lip-Sync:
- `mediapipe` — Facial detection/landmarks
- `pytorch-lightning` — Training framework (optional)

### Audio:
- `librosa` — Audio manipulation
- `julius` — High-quality resampling
- `torch-audiomentations` — Augmentation

### Color:
- `OpenColorIO` — Color space management
- `ACES` OCIO config — Professional color pipeline
- `imageio` — Image I/O with EXR support

### Video:
- `FFmpeg` bindings (python-av, opencv-python)
- `imageio-ffmpeg` — FFmpeg backend

### Editing:
- `xml.etree` — FCPXML generation
- Custom EDL/AAF writers

---

## KPIs & Success Metrics

### Step 2.1 (TTS):
- [ ] MOS score ≥ 4.0
- [ ] < 100ms latency per second of audio
- [ ] ≥ 7 languages supported
- [ ] ≥ 10 unique speaker voices

### Step 2.2 (Lip-Sync):
- [ ] LSE-D ≤ 7.0
- [ ] LSE-C ≥ 6.0
- [ ] 30 fps real-time on GPU
- [ ] Tested on ≥ 5 facial types

### Step 2.3 (Audio Mixer):
- [ ] - 14 LUFS ± 0.5 consistency
- [ ] < 0.1% clipping distortion
- [ ] 16+ track mixing capability
- [ ] 5.1 surround tested

### Step 2.4 (Montage):
- [ ] Timeline exports to EDL/FCPXML/AAF
- [ ] Pacing ±5% of target durations
- [ ] ≥ 5 transition types working
- [ ] Continuity check ≥ 90% accuracy

### Step 2.5 (Color):
- [ ] ≥ 20 LUTs available
- [ ] HDR10 export metadata correct
- [ ] Scene color matching ±ΔE 3.0
- [ ] Auto-grade produces rated 4.0+ videos

### Step 2.6 (Export):
- [ ] H.265, ProRes, DNxHR verified playable
- [ ] Bitrate within ±5% of spec
- [ ] Audio sync ≤ 23ms offset (imperceptible)
- [ ] 4 format export tested

### Step 2.7 (Nodes):
- [ ] All 5 nodes call real computation
- [ ] End-to-end inference completes
- [ ] 3x inference speedup vs Phase 1 mocks
- [ ] 0 instances of torch.randn() in critical nodes

---

## Risk Mitigation

### Known Risks:

| Risk | Mitigation | Owner |
|------|-----------|-------|
| TTS quality below 4.0 MOS | Fine-tune on diverse speakers; A/B test | TTS Engineer |
| Lip-sync drift over long videos | Implement frame-wise correction; test ≥ 60s videos | LipSync Engineer |
| Audio loudness inconsistency | Integrate loudness metering; A/B test before export | Audio Engineer |
| Pacing produces unnatural timing | Test with ≥ 5 human reviewers; iterate algorithm | Editor |
| HDR tone mapping oversaturates colors | Use reference cinema content; grade comparison | Colorist |
| Incompatible export formats | Test in Premiere, Avid, Final Cut Pro | VFX Pipeline Eng |
| Inference performance degradation | Benchmark memory usage; optimize bottlenecks | Infra Engineer |

---

## Timeline & Milestones

| Milestone | Week(s) | Status | Owner |
|-----------|---------|--------|-------|
| TTS model trained & evaluated | W1-3 | 🔴 TO START | [TTS] |
| Lip-sync model inference working | W4-6 | 🔴 TO START | [LipSync] |
| Audio mixer real-time @ 16 tracks | W7-8 | 🔴 TO START | [Audio] |
| Montage engine produces EDL | W9-10 | 🔴 TO START | [Editing] |
| Color grading with 20+ LUTs | W11-12 | 🔴 TO START | [Color] |
| Multi-format export tested | W13-14 | 🔴 TO START | [Export] |
| All inference nodes connected | W15-16 | 🔴 TO START | [Infra] |
| **PHASE 2 COMPLETE** | **W16** | 🔴 PENDING | All |

---

## Getting Started

### For TTS Engineer:
1. Create `TextFrontend` class with G2P and normalization
2. Implement `ProsodyModeler` with F0 and duration prediction
3. Train on LibriTTS (or fine-tune StyleTTS2)
4. Evaluate MOS score

### For Lip-Sync Engineer:
1. Prepare LRS2 2.0 dataset (audio + video + FLAME params)
2. Implement audio encoder + temporal processor
3. Train with LSE-D loss
4. Benchmark LSE-D, LSE-C on test set

### For Audio Engineer:
1. Implement compressor, EQ, reverb dynamics
2. Test LUFS normalization (-14 target)
3. Implement 5.1 spatial routing
4. Integrate with TTS output

### For Editor:
1. Parse scenario JSON structure
2. Implement temporal positioning algorithm
3. Define transition prevalence by emotion type
4. Test pacing on ≥ 10 sample scripts

### For Colorist:
1. Load 20+ industry-standard LUTs (.cube format)
2. Implement 3D LUT interpolation
3. Train auto-grader on cinema footage
4. Validate color matching across scenes

### For Export Engineer:
1. Profile FFmpeg- wrappers for video/audio encoding
2. Implement ExportProfile system
3. Test muxing in professional NLEs
4. Benchmark quality/filesize tradeoffs

### For Infra Engineer:
1. Identify all mock nodes in inference graph
2. Create integration adapters
3. Benchmark real inference (memory, latency)
4. Performance optimize critical paths

---

## Deliverables Checklist

### Code:
- [ ] All 6 module directories created with stubs
- [ ] 20+ implementation files (model.py, etc.)
- [ ] Comprehensive docstrings and TODOs
- [ ] Unit tests for each component
- [ ] Integration test for full pipeline

### Documentation:
- [ ] Step 2.1-2.7 completion reports (one per week)
- [ ] API documentation for each module
- [ ] Configuration & tuning guides
- [ ] Troubleshooting guide

### Data & Models:
- [ ] TTS  model weights (StyleTTS2 fine-tuned)
- [ ] Lip-sync model weights (trained on LRS2)
- [ ] 20+ color grading LUTs
- [ ] Pre-computed ACES OCIO config

### Testing:
- [ ] ≥ 150 unit tests (20-30 per step)
- [ ] All tests passing
- [ ] Coverage ≥ 70% on critical modules
- [ ] Integration test suite passing

### Integration:
- [ ] All Phase 2 modules integrated
- [ ] Inference nodes fully connected  
- [ ] End-to-end pipeline functional
- [ ] Performance ≥ Spec targets

---

## Next: Phase 3 (Weeks 25-36)

Once Phase 2 completes:
- **Phase 3:** Production infrastructure
  - Distributed training
  - Model versioning (MLflow)
  - Monitoring & observability
  - GPU fleet management

---

## Contact & Questions

Phase 2 is **now live**. Work began February 14, 2026.

Target Completion: **May 30, 2026** ✅

Let's build the cinematic pipeline!

---

**Document Created:** February 14, 2026  
**Phase Status:** 🚀 LAUNCHED  
**Next Milestone:** TTS Module completion (Week 3)
