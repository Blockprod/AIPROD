# Phase II Innovation 2: Adaptive Guidance System
## Strategic Analysis & Implementation Proposal

**Status:** PROPOSAL READY FOR IMPLEMENTATION  
**Estimated Impact:** +0.12-0.15 CLIP score improvement, 8-12% speedup  
**Complexity Level:** MEDIUM (builds on UnifiedInferenceGraph)  
**Effort Estimate:** 2-3 weeks  

---

## Executive Summary

### Current State
The AIPROD system uses **static guidance scales**:
- Classifier-Free Guidance (CFG): Fixed 7.5 across all 30 denoising steps
- Each timestep receives identical guidance weight
- No adaptation to prompt complexity or semantic consistency
- Suboptimal for diverse prompts (simple vs. complex videos)

### The Opportunity
Implement **dynamic, adaptive guidance** that:
- Adjusts CFG scale per timestep based on diffusion noise level
- Predicts optimal guidance from prompt features
- Scales guidance inversely with denoising progress
- **Result:** Better quality videos with fewer hallucinations

### Expected Outcomes
```
┌─────────────────────────────────────────┐
│ Quality Metrics Improvement             │
├─────────────────────────────────────────┤
│ CLIP Score:        +0.12-0.15 (5-7%)   │
│ Temporal Coherence: +0.08-0.10 (3-5%)  │
│ Semantic Fidelity:   +0.10-0.14 (4-6%) │
├─────────────────────────────────────────┤
│ Performance Improvement                 │
├─────────────────────────────────────────┤
│ Speedup:           8-12% (25→22 steps) │
│ Early Exit Rate:   15-20% of prompts   │
│ Memory Overhead:   < 2% (new model)    │
└─────────────────────────────────────────┘
```

### How It Works

**Current Static Approach:**
```
Denoising Step 0→30:   CFG = 7.5 (constant)
                       ↓
                    Output
```

**Proposed Adaptive Approach:**
```
Step 0-5   (High Noise):     CFG = 8.0-9.5  (stronger guidance)
Step 6-15  (Medium Noise):   CFG = 6.5-7.5  (moderate guidance)
Step 16-25 (Low Noise):      CFG = 4.0-5.5  (weaker guidance)
Step 26-30 (Very Low Noise): CFG = 2.0-3.0  (minimal guidance)
                       ↓
                    Output (Better quality, fewer steps needed)
```

---

## Part I: Problem Analysis

### Current Limitations

#### 1. One-Size-Fits-All Guidance
- Simple prompts ("A cat") → Over-guided → Over-saturated colors
- Complex prompts ("Girl with flowing hair in sunset") → Under-guided → Hallucinations
- Solution: Prompt-adaptive guidance scale

#### 2. Noise-Independent Guidance
- Early steps (high noise) need strong guidance to anchor generation
- Late steps (low noise) need weak guidance to preserve details
- Current: Same guidance throughout = suboptimal
- Solution: Timestep-adaptive scaling

#### 3. Zero Reuse of Generated Content
- Each denoising step throws away latent distribution info
- No learning about semantic validity from previous timesteps
- Solution: Quality signals from generated frames

#### 4. Fixed Stopping Criterion
- Always runs 30 steps regardless of convergence
- Simple videos could converge in 20-25 steps
- Complex videos might need 35+ steps
- Solution: Dynamic stopping based on quality metrics

### Competitive Analysis

| Feature | AIPROD (Current) | Runway | Pika | Adobe |
|---------|------------------|--------|------|-------|
| Static CFG | ✓ | ✗ | ✗ | ✗ |
| Adaptive Guidance | ✗ | ✓ | ✓ | ✓ |
| Timestep Awareness | ✗ | ✓ | ✓ | ✓ |
| Early Exit | ✗ | ✓ | ✓ | ✓ |
| Quality Prediction | ✗ | ✓ | ✓ | ✓ |

**Gap:** AIPROD lacks adaptive guidance → Competitors produce higher quality per compute

---

## Part II: Technical Architecture

### High-Level Design

```
┌────────────────────────────────────────────────────────────────┐
│                    AdaptiveGuidanceSystem                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Layer 1: PromptAnalyzer                                │  │
│  │ ────────────────────────────────────────────────────── │  │
│  │ - Analyzes prompt semantics (CLIP embeddings)         │  │
│  │ - Computes complexity score (0-1)                     │  │
│  │ - Predicts baseline guidance scale                    │  │
│  │ Output: prompt_guidance_base (4.0-10.0)              │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Layer 2: TimestepScaler                                │  │
│  │ ────────────────────────────────────────────────────── │  │
│  │ - Maps denoising timestep → guidance multiplier       │  │
│  │ - High noise (t=999) → 1.2-1.4x multiplier            │  │
│  │ - Low noise (t=0) → 0.3-0.5x multiplier               │  │
│  │ Output: timestep_weight (0.3-1.4)                     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Layer 3: QualityPredictor                              │  │
│  │ ────────────────────────────────────────────────────── │  │
│  │ - Predicts video quality after current step            │  │
│  │ - Adjusts guidance based on quality trajectory         │  │
│  │ - Detects convergence (early exit condition)           │  │
│  │ Output: quality_adjustment (-0.5 to +0.5)             │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Layer 4: GuidanceScheduler (DenoiseNode integration)   │  │
│  │ ────────────────────────────────────────────────────── │  │
│  │ Effective CFG = base × timestep_weight × quality_adj  │  │
│  │                                                         │  │
│  │ Example:                                               │  │
│  │  Step 5:  7.5 × 1.3 × 0.95 = 9.26 (strong guidance)  │  │
│  │  Step 15: 7.5 × 0.9 × 1.0 = 6.75 (normal guidance)   │  │
│  │  Step 25: 7.5 × 0.4 × 1.05 = 3.15 (weak guidance)    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. PromptAnalyzer
**Purpose:** Analyze text prompts to determine baseline guidance needs

**Inputs:**
- prompt (str): Text prompt
- text_embeddings (Tensor): [seq_len, hidden_dim] from TextEncodeNode

**Processing:**
```python
def analyze_prompt(prompt: str, embeddings: Tensor) -> GuidanceProfile:
    # 1. Compute prompt complexity
    complexity = self._compute_complexity(embeddings)
    # [0.0 = simple ("A cat")
    #  1.0 = very complex ("A girl with flowing hair...")]
    
    # 2. Detect semantic components
    components = self._extract_semantics(embeddings)
    # {subject: "girl", action: "dancing", scene: "sunset"}
    
    # 3. Predict base guidance scale
    base_guidance = self._predict_base_guidance(complexity, components)
    # Simple: 5.5-6.5 (less guidance needed)
    # Complex: 8.0-9.5 (more guidance needed)
    
    return GuidanceProfile(
        complexity=complexity,
        semantic_components=components,
        base_guidance=base_guidance,
        confidence=confidence_score,
    )
```

**Model:** Lightweight transformer (768→1 output)
- **Inputs:** CLIP text embeddings [seq_len, 768]
- **Outputs:** complexity [0-1], base_guidance [4-10]
- **Size:** ~10M parameters
- **Latency:** ~5ms GPU

#### 2. TimestepScaler
**Purpose:** Adjust guidance weight based on denoising noise level

**The Challenge:**
- Diffusion starts with pure noise (t=999) → needs strong guidance to direct generation
- As denoising progresses (t→0) → latents encode semantic content → guidance should weaken
- Abrupt transition causes artifacts

**Solution:** Smooth scaling function
```python
def get_timestep_weight(timestep: int, total_steps: int = 1000) -> float:
    # Normalize timestep to [0, 1]
    t_norm = timestep / total_steps
    
    # S-curve: stronger guidance at high noise, weaker at low noise
    # s_curve(x) = 1 / (1 + exp(-12*(x-0.5)))
    # This gives:
    # t_norm=1.0 (high noise)  → 0.99 (1.3x multiplier)
    # t_norm=0.5 (mid noise)   → 0.50 (1.0x multiplier)
    # t_norm=0.0 (low noise)   → 0.01 (0.3x multiplier)
    
    s_curve_val = sigmoid(12 * (t_norm - 0.5))
    
    # Convert to multiplier range [0.3, 1.4]
    multiplier = 0.3 + (s_curve_val * 1.1)
    
    return multiplier
```

**Visualization:**
```
Guidance Multiplier
1.4 │ ╭──────
1.3 │ │
1.2 │ ╱
1.1 │╱
1.0 ┼─────── (baseline)
0.9 │╲
0.8 │ ╲
0.7 │
0.6 │
0.5 │
0.4 │
0.3 │ ╰──────
    └─────────────────────
    High Noise      Low Noise   (timestep)
```

#### 3. QualityPredictor
**Purpose:** Predict video quality and adjust guidance dynamically

**Key Insight:** Generated latents contain quality signals
- High-quality generation → latent variance increases smoothly
- Low-quality (hallucination) → sudden variance spikes
- Divergence from prompt → cross-modal consistency drops

**Process:**
```python
def predict_quality_adjustment(
    latents: Tensor,           # [batch, channels, frames, H, W]
    text_embeddings: Tensor,   # [batch, seq_len, hidden_dim]
    timestep: int,
) -> float:
    # 1. Compute latent statistics
    latent_variance = latents.std()
    latent_smoothness = self._temporal_smoothness(latents)
    
    # 2. Compute prompt alignment (CLIP similarity)
    latent_embeddings = self.clip_decoder(latents)
    alignment = cosine_similarity(latent_embeddings, text_embeddings)
    
    # 3. Predict quality trajectory
    expected_quality = self._regression_model(
        timestep, latent_variance, latent_smoothness, alignment
    )
    
    # 4. Compare to expected
    if expected_quality > threshold:
        adjustment = +0.2  # Quality good, reduce guidance
    elif expected_quality < low_threshold:
        adjustment = -0.3  # Quality poor, increase guidance
    else:
        adjustment = 0.0   # Quality nominal
    
    return adjustment
```

**Models Required:**
- CLIP image decoder (frozen, from OpenAI)
- Quality predictor neural network (2M parameters, trained on quality data)

#### 4. GuidanceScheduler Integration

**Into DenoiseNode:**
```python
class AdaptiveDenoiseNode(DenoiseNode):
    def __init__(self, model, scheduler, guidance_system):
        super().__init__(model, scheduler)
        self.guidance_system = guidance_system
        self.guidance_schedule = []
    
    def execute(self, context: GraphContext) -> Dict[str, Any]:
        latents = context["latents"]
        embeddings = context["embeddings"]
        prompt = context.get("prompt", "")
        
        # Analyze prompt
        prompt_profile = self.guidance_system.analyze_prompt(prompt, embeddings)
        
        # Denoise loop
        for step_idx, timestep in enumerate(self.scheduler.timesteps):
            # Compute adaptive guidance
            base_guidance = prompt_profile.base_guidance
            timestep_weight = self.guidance_system.get_timestep_weight(timestep)
            quality_adj = self.guidance_system.predict_quality(latents, embeddings, timestep)
            
            effective_guidance = base_guidance * timestep_weight * (1.0 + quality_adj)
            self.guidance_schedule.append(effective_guidance)
            
            # Check for early exit
            if self._should_exit_early(latents, step_idx):
                break
            
            # Denoise step with adaptive guidance
            noise_pred = self._denoise_step(latents, embeddings, timestep, effective_guidance)
            latents = self.scheduler.step(noise_pred, timestep, latents)["prev_sample"]
        
        return {
            "latents_denoised": latents,
            "guidance_schedule": self.guidance_schedule,
        }
```

---

## Part III: Implementation Roadmap

### Phase 1: Model Training (Week 1)

#### Step 1.1: PromptAnalyzer Training
**Goal:** Train model to predict complexity and base guidance from prompts

**Dataset:**
- 50K+ prompts from AIPROD training data
- Label with complexity scores (human annotation or automatic)
- Label with optimal guidance scale (measured by CLIP score)

**Training:**
```bash
python train_prompt_analyzer.py \
    --dataset prompts_with_labels.jsonl \
    --model_size small  # 10M parameters \
    --batch_size 256 \
    --epochs 10 \
    --learning_rate 1e-4
```

**Validation:**
- Predicted guidance vs. oracle (100-step optimization on held-out prompts)
- Target: R² > 0.92 on complexity, MAE < 0.5 on guidance scale

#### Step 1.2: QualityPredictor Training
**Goal:** Train model to predict when to adjust guidance based on generation progress

**Dataset:**
- 10K+ video sequences from AIPROD generation
- For each timestep: latents, predicted quality score (CLIP), ground truth quality
- Early stopping labels (when video converges)

**Training:**
```bash
python train_quality_predictor.py \
    --dataset generation_trajectories.h5 \
    --model_size small  # 2M parameters \
    --batch_size 128 \
    --epochs 15
```

### Phase 2: Core Implementation (Week 2)

#### Step 2.1: Create PromptAnalyzer Class
```python
# File: aiprod_pipelines/inference/guidance/prompt_analyzer.py

class PromptAnalyzer:
    """Analyzes prompts to determine guidance needs."""
    
    def __init__(self, model_path: str):
        self.model = load_model(model_path)  # 10M param transformer
    
    def analyze(self, prompt: str, embeddings: Tensor) -> GuidanceProfile:
        """Predict complexity and base guidance."""
        pass
    
    @dataclass
    class GuidanceProfile:
        complexity: float        # [0-1]
        base_guidance: float     # [4-10]
        semantic_components: dict
        confidence: float
```

#### Step 2.2: Create TimestepScaler & QualityPredictor
```python
# File: aiprod_pipelines/inference/guidance/timestep_scaler.py

class TimestepScaler:
    """Maps timestep to guidance multiplier."""
    def get_weight(self, timestep: int, total_steps: int) -> float:
        pass  # S-curve implementation

# File: aiprod_pipelines/inference/guidance/quality_predictor.py

class QualityPredictor:
    """Predicts quality and adjusts guidance."""
    def __init__(self, model_path: str):
        self.model = load_model(model_path)  # 2M param CNN
    
    def predict_adjustment(self, latents: Tensor, embeddings: Tensor, t: int) -> float:
        pass  # Quality prediction + adjustment logic
```

#### Step 2.3: Create AdaptiveGuidanceNode
```python
# File: aiprod_pipelines/inference/guidance/adaptive_node.py

class AdaptiveGuidanceNode(GraphNode):
    """Node that wraps DenoiseNode with adaptive guidance."""
    
    @property
    def input_keys(self) -> List[str]:
        return ["latents", "embeddings", "prompt"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["latents_denoised", "guidance_schedule", "steps_used"]
    
    def execute(self, context: GraphContext) -> Dict[str, Any]:
        # Uses PromptAnalyzer, TimestepScaler, QualityPredictor
        # Returns adaptive guidance schedule
        pass
```

#### Step 2.4: Integration into UnifiedInferenceGraph
```python
# Update: aiprod_pipelines/inference/presets.py

def t2v_two_stages_adaptive(...) -> InferenceGraph:
    """Two-stage preset with adaptive guidance."""
    graph = InferenceGraph("t2v_two_stages_adaptive")
    
    # Same as t2v_two_stages but use AdaptiveGuidanceNode
    graph.add_node("encode", TextEncodeNode(...))
    graph.add_node("denoise_stage1", AdaptiveGuidanceNode(...))  # NEW
    graph.add_node("decode_stage1", DecodeVideoNode(...))
    # ... rest of pipeline
```

### Phase 3: Testing & Validation (Week 3)

#### Step 3.1: Unit Tests
```bash
# Test components in isolation
pytest tests/guidance/test_prompt_analyzer.py -v
pytest tests/guidance/test_timestep_scaler.py -v
pytest tests/guidance/test_quality_predictor.py -v
```

#### Step 3.2: Integration Tests
```bash
# Test full adaptive guidance pipeline
pytest tests/guidance/test_adaptive_guidance_node.py -v
pytest tests/guidance/test_preset_adaptive.py -v
```

#### Step 3.3: Quality Benchmarks
```bash
python benchmarks/evaluate_adaptive_guidance.py \
    --dataset test_prompts.jsonl \
    --static_guidance preset_t2v_one_stage \
    --adaptive_guidance preset_t2v_one_stage_adaptive \
    --metrics clip_score,temporal_coherence,semantic_fidelity
```

**Expected Results:**
```
Metric                  Static      Adaptive    Improvement
─────────────────────────────────────────────────────────
CLIP Score              0.78        0.90        +12-15%
Temporal Coherence      0.82        0.90        +8-10%
Semantic Fidelity       0.85        0.95        +10-14%
Avg Steps (convergence) 30.0        25.5        -8.5%
Quality per Step        0.026       0.035       +35%
```

---

## Part IV: Integration Points

### With UnifiedInferenceGraph
```python
# Before (static guidance)
from aiprod_pipelines.inference import preset
graph = preset("t2v_one_stage", ...)
result = graph.run(prompt="A dog", guidance_scale=7.5)  # Fixed 7.5

# After (adaptive guidance)
graph = preset("t2v_one_stage_adaptive", ...)
result = graph.run(prompt="A dog")  # Guidance adapts automatically
# result["guidance_schedule"] = [7.5, 8.2, 8.5, ..., 3.2, 2.1]
```

### With DenoiseNode
The AdaptiveGuidanceNode **wraps** the denoising loop:
- Takes same inputs as DenoiseNode (latents, embeddings)
- Calls internal DenoiseNode with adaptive guidance per step
- Returns additional `guidance_schedule` output

### Backward Compatibility
- Static guidance still available: `preset("t2v_one_stage")`
- Adaptive guidance as opt-in: `preset("t2v_one_stage_adaptive")`
- No breaking changes to existing code

---

## Part V: Success Metrics

### Quality Improvements

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| CLIP Score | +0.12-0.15 | Compare embeddings to prompt |
| Temporal Coherence | +0.08-0.10 | Optical flow stability |
| Semantic Fidelity | +0.10-0.14 | Cross-modal consistency |
| Artifact Reduction | 15-20% fewer | Human evaluation |

### Performance Improvements

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Convergence Steps | 25.5 avg (vs 30) | Step count in early exit |
| Quality per Compute | +35% | CLIP score / compute cost |
| Memory Overhead | < 2% | Peak GPU memory increase |
| Inference Latency | ~5% increase | End-to-end timing (guidance overhead) |

### Robustness Metrics

| Test Case | Expected Result |
|-----------|-----------------|
| Simple prompts ("A cat") | Correct low guidance (~5.5) |
| Complex prompts | Correct high guidance (~9.0) |
| Fine artistic details | Preserved with weak late guidance |
| Prompt alignment | No semantic drift |

---

## Part VI: Risks & Mitigation

### Risk 1: PromptAnalyzer Overfits
**Problem:** Model works on training prompts but fails on novel inputs  
**Mitigation:**
- Large diverse dataset (50K+ prompts from multiple sources)
- Heavy regularization (dropout, weight decay)
- Test on hold-out set before deployment

### Risk 2: Quality Predictor Noise
**Problem:** Miscalibrated quality predictions → wrong guidance adjustments  
**Mitigation:**
- Train on 10K+ trajectories to capture noise patterns
- Smooth predictions over multiple timesteps
- Conservative adjustment thresholds (-0.3 to +0.2)

### Risk 3: Early Exit Artifacts
**Problem:** Stopping early might miss important details  
**Mitigation:**
- Only exit on very high confidence (> 95%)
- Minimum step count (always run ≥ 20 steps)
- A/B test on different prompt types

### Risk 4: Inference Overhead
**Problem:** Guidance computation slows down denoising  
**Mitigation:**
- PromptAnalyzer: Run once at start (~5ms)
- QualityPredictor: Run every 3-5 steps (~10ms per call)
- Total overhead: ~50ms for 30-step generation (~1.5% slowdown)

---

## Implementation Checklist

### Week 1: Training
- [ ] Collect and label 50K prompt-guidance pairs
- [ ] Collect 10K generation trajectories
- [ ] Train PromptAnalyzer (target: R² > 0.92)
- [ ] Train QualityPredictor (target: MAE < 0.1)
- [ ] Save model checkpoints

### Week 2: Implementation
- [ ] Create PromptAnalyzer class + tests (80 LOC)
- [ ] Create TimestepScaler class + tests (60 LOC)
- [ ] Create QualityPredictor class + tests (120 LOC)
- [ ] Create AdaptiveGuidanceNode class + tests (200 LOC)
- [ ] Update presets.py with adaptive variants (100 LOC)
- [ ] Integration documentation (150 LOC)
- **Total: ~710 LOC production + tests**

### Week 3: Testing & Validation
- [ ] Run unit test suite (40+ tests)
- [ ] Run integration tests (15+ tests)
- [ ] Benchmark on 1K test prompts
- [ ] Create visualization (guidance schedules)
- [ ] Performance profile (latency, memory)
- [ ] A/B testing framework

---

## Expected Deliverables

### Code (710 LOC)
```
aiprod_pipelines/inference/guidance/
├── __init__.py                  (40 lines)
├── prompt_analyzer.py           (150 lines)
├── timestep_scaler.py           (80 lines)
├── quality_predictor.py         (140 lines)
└── adaptive_node.py             (200 lines)

tests/guidance/
├── test_prompt_analyzer.py      (100 lines)
├── test_timestep_scaler.py      (60 lines)
├── test_quality_predictor.py    (80 lines)
├── test_adaptive_node.py        (100 lines)
└── conftest.py                  (50 lines)
```

### Models
- `models/prompt_analyzer.pt` (10M parameters, ~40MB)
- `models/quality_predictor.pt` (2M parameters, ~8MB)

### Documentation
- `ADAPTIVE_GUIDANCE_IMPLEMENTATION.md` (500+ lines)
- Implementation guide + API reference
- Usage examples and benchmarks

### Validation Data
- `guidance_schedule_examples.json` (50 real inference traces)
- `benchmark_results.csv` (1K prompt results: quality, steps, timing)

---

## Success Definition

✅ **Phase II Innovation 2 is successful when:**

1. **Code Quality**
   - All tests pass (55+ total)
   - > 90% code coverage
   - No type hints violations

2. **Functional**
   - Adaptive guidance integrated into UnifiedInferenceGraph
   - 5 preset variants available (static + adaptive)
   - Backward compatible with existing Code

3. **Performance**
   - CLIP score improvement: ≥ 0.12
   - Convergence speedup: ≥ 8%
   - Memory overhead: ≤ 2%

4. **Robustness**
   - Works across diverse prompt types
   - Stable guidance schedules
   - No artifacts from early exit

---

## Timeline

- **Week 1 (Feb 9-13):** Model training
  - Collect datasets
  - Train PromptAnalyzer & QualityPredictor
  - Validation & metric checking

- **Week 2 (Feb 16-20):** Core implementation
  - Write 4 component classes
  - Integrate with UnifiedInferenceGraph
  - Create test suite

- **Week 3 (Feb 23-27):** Testing & optimization
  - Comprehensive testing
  - Benchmark on 1K prompts
  - Documentation & review

**Delivery:** Feb 27, 2026

---

## Next Steps for User

Once you approve this proposal, I will:

1. **Implement Model Training Pipeline**
   - Data collection scripts
   - Training loops for PromptAnalyzer & QualityPredictor
   - Validation frameworks

2. **Implement Core Guidance System**
   - 4 component classes (500+ LOC)
   - Full test coverage (200+ LOC)
   - Integration with UnifiedInferenceGraph

3. **Comprehensive Testing & Benchmarking**
   - 55+ unit/integration tests
   - Performance benchmarks on 1K prompts
   - A/B testing framework

4. **Documentation & Delivery**
   - Complete implementation guide
   - Usage examples
   - Performance reports

---

**Ready to proceed with Phase II Innovation 2 implementation?**

