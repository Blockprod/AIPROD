# AIPROD Data Governance Policy

**Version:** 1.0  
**Date:** February 14, 2026  
**Status:** Active  
**Owner:** AIPROD Core Team

---

## 1. Overview & Purpose

This document establishes the governance framework for AIPROD's dataset pipeline, covering:
- **Data sourcing** — Acceptable sources and licensing
- **Data quality standards** — Resolution, duration, audio requirements
- **Data versioning** — Tracking changes and reproducibility
- **Data privacy & compliance** — GDPR, CCPA, content rights
- **Data lineage** — Source → processing → training attribution

---

## 2. Data Sourcing Policy

### 2.1 Acceptable Video Sources

#### Licensed Public Datasets
| Source | License | Types | Usage |
|--------|---------|-------|-------|
| **Kinetics-700** | CC-BY | Human action, sports, activities | ✅ Fine-tuning, quality baseline |
| **Activitynet** | CC-BY 3.0 | Event videos, temporal localization | ✅ Temporal modeling |
| **Something-Something v2** | CC-BY 4.0 | Human-object interaction | ✅ Object interaction modeling |
| **MSR-VTT** | MSRVTT License | Video-text pairs | ✅ Caption-video alignment |
| **YouCook2** | YouTube CC-BY | Instructional cooking videos | ✅ Procedural understanding |
| **WebVid** | CC-BY 4.0 | Short-form web videos | ✅ Natural video generation |
| **LAION Video** | CC-BY 4.0 | Large-scale video-text | ✅ Pretraining corpus |

#### Commercial Licenses (Priority)
- **Shutterstock Video License** — Commercial usage rights, 4K quality
- **Getty Images Video License** — Editorial + commercial, curated
- **Pond5** — Royalty-free, diverse content
- **iStock by Getty Images** — Extended commercial license available

#### Generated/Synthetic (Approved)
- **CGI datasets** — Blender 3D rendered scenes (owned, no licensing issue)
- **Game footage** — Licensed games with commercial usage rights
- **Synthetic dialog** — AI-generated video + custom voiceover

### 2.2 Sources to AVOID

| Category | Reason | Alternative |
|----------|--------|-------------|
| YouTube (arbitrary) | Copyright risk, mixed licensing | Use YouTube with explicit CC-BY videos only |
| TikTok/Instagram | EULA restricts commercial training | Use licensed alternatives |
| Netflix/Disney content | Copyrighted, not licensed for AI training | Use authorized datasets instead |
| Facebook/Instagram UGC | Terms of service prohibit AI training | Use user-opted-in datasets |
| Pirated content | Legal risk + ethical issues | 🚫 Never |

---

## 3. [Data Quality Standards](#data-quality-standards)

### 3.1 Video Specifications

```yaml
Resolution:
  minimum: 480p (854×480)
  target: 1080p (1920×1080)
  max_supported: 4K (3840×2160)
  
Framerate:
  minimum: 24 fps
  standard: 30 fps
  max: 60 fps
  note: "Interpolate to 30fps if needed"

Duration:
  minimum: 2 seconds
  target: 5-60 seconds
  maximum: 600 seconds (10 minutes)
  
Bitrate:
  minimum: 2 Mbps
  target: 8-15 Mbps
  maximum: 50 Mbps

Codec:
  supported:
    - H.264
    - H.265/HEVC
    - VP9
    - ProRes (for high-quality ingestion)

Color Space:
  required: sRGB or rec709
  bit_depth: 8-bit minimum (10-bit preferred for quality)
```

### 3.2 Audio Specifications (if available)

```yaml
Channels:
  mono: 1
  stereo: 2
  surround: ✅ accepted but downmixed to stereo

Sample Rate:
  minimum: 16 kHz
  standard: 44.1 kHz or 48 kHz
  maximum: 192 kHz (resampled to 48 kHz)

Bitrate:
  minimum: 128 kbps
  target: 256 kbps
  maximum: 320 kbps

Formats:
  - MP3
  - AAC
  - WAV (PCM 16-bit or 24-bit)
  - FLAC
```

### 3.3 Content Requirements

#### Minimum Acceptable Content
- ✅ Diverse camera movements (pan, tilt, zoom, dolly)
- ✅ Natural lighting conditions or well-lit scenes
- ✅ Clear visual foreground/background separation
- ✅ Audio synced to video (±100ms tolerance)
- ✅ No watermarks, logos, or hardcoded text >10% of frame

#### Automatically REJECTED
- 🚫 Black screens or static scenes >2 seconds
- 🚫 Completely dark or overexposed frames >20%
- 🚫 Heavy compression artifacts or banding
- 🚫 Aspect ratio <4:3 or >21:9
- 🚫 NSFW content (violence, explicit content)
- 🚫 Vertically-oriented mobile video (except explicitly for mobile)

#### Manual Review Required
- ⚠️ Frame rate inconsistency
- ⚠️ Audio quality issues (clipping, extreme noise)
- ⚠️ Copyright/watermark presence

---

## 4. Data Classification & Annotation

### 4.1 Metadata Requirements

All videos MUST include:

```json
{
  "id": "unique_identifier_v1",
  "source": "dataset_name",
  "license": "CC-BY-4.0 | Commercial | Proprietary",
  "download_date": "2026-02-14",
  "video_path": "s3://bucket/path/video.mp4",
  
  "technical_metadata": {
    "duration_seconds": 15.2,
    "resolution": "1920×1080",
    "fps": 30,
    "bitrate_kbps": 8000,
    "codec": "h264",
    "color_space": "rec709",
    "has_audio": true,
    "audio_language": "en",
    "audio_channels": 2
  },
  
  "content_metadata": {
    "caption": "Person dancing in a bright room",
    "scene_description": "Indoor, modern apartment, natural lighting",
    "main_subjects": ["person", "dance"],
    "camera_movement": ["pan", "zoom"],
    "tags": ["dance", "indoor", "motion"]
  },
  
  "quality_score": {
    "visual_quality": 0.85,
    "audio_quality": 0.80,
    "relevance": 0.88,
    "overall": 0.84
  },
  
  "processing_status": {
    "downloaded": true,
    "validated": true,
    "scene_split": true,
    "captioned": true,
    "latents_computed": false
  }
}
```

### 4.2 Content Tagging Schema

```yaml
Primary Categories:
  - human_action: Person performing activity
  - object_interaction: Manipulation of objects
  - nature_scene: Landscapes, animals, natural phenomena
  - synthetic: CGI, animated, or artificial content
  - mixed: Combination of above

Sub-categories (examples):
  human_action:
    - dance | sports | exercise | music | performance
  object_interaction:
    - cooking | building | gaming | crafting | sports_equipment
  nature_scene:
    - outdoor | animals | weather | landscape
  synthetic:
    - 3d_render | animation | motion_graphics
    
Camera Movement:
  - static
  - pan: horizontal camera movement
  - tilt: vertical camera movement
  - zoom: focal length change
  - dolly: camera position change
  - crane: complex 3D camera motion
  
Lighting:
  - natural: daylight, sunlight
  - artificial: indoor lights, studio
  - mixed: combination
  - challenging: very dark, backlit, or extreme
  
Audio Quality:
  - perfect: broadcast quality
  - good: minimal background noise
  - acceptable: some background noise tolerable
  - degraded: significant artifacts
```

---

## 5. Data Versioning & Tracking

### 5.1 DVC (Data Version Control) Setup

All dataset versions tracked via DVC with S3 backend:

```bash
# Dataset versioning structure
datasets/
├── v1/
│   ├── raw/                      # Original downloads
│   ├── processed/                # After validation + scene split
│   ├── captions/                 # Generated captions
│   ├── metadata.dvc              # DVC tracking
│   └── manifest.json             # Complete inventory
├── v2/
│   └── ...
└── latest → v1/                  # Symlink to current version
```

### 5.2 Version Changelog

```yaml
v1.0:
  date: 2026-02-01
  sources:
    - Something-Something v2: 100.2 hours
    - WebVid subset: 50.1 hours
    - Custom licensed: 12.8 hours
  total_hours: 163.1
  videos: 12,847
  qc_passed: 12,623 (98.2%)
  changes:
    - Initial dataset creation
    - Quality filtering at 85% threshold
    - Kinetics-700 subset integration
  
v1.1:
  date: 2026-02-14
  total_hours: 201.5
  videos: 15,923
  changes:
    - Added Activitynet subset
    - Improved caption quality
    - Audio quality filtering stricter

v2.0:
  planned: 2026-03-15
  expected_hours: 500+
  planned_sources:
    - LAION Video (first 100K samples)
    - Getty Images commercial license
    - Custom curated collection
```

### 5.3 Metadata Tracking

```bash
# Every dataset version includes:
datasets/v1/
├── manifest.json          # Complete file inventory + hashes
├── statistics.json        # Duration, resolution, codec distributions
├── quality_report.csv     # Per-video QC scores
├── changelog.md           # What changed vs previous version
└── .gitignore             # Keep only metadata, not raw videos
```

---

## 6. Data Privacy & GDPR Compliance

### 6.1 Privacy Requirements

**All videos MUST NOT contain:**
- 🚫 Identifiable faces from non-celebrities (unless explicit consent)
- 🚫 Personal information (names, address, ID numbers)
- 🚫 Biometric data that could identify individuals
- 🚫 Private health/medical information

**Whitelisted exceptions:**
- ✅ Celebrity/public figure faces (>500K followers or Wikipedia entry)
- ✅ Generic crowd scenes (>50 people, not individually identifiable)
- ✅ Blurred or anonymized faces

### 6.2 Consent Recording

For any dataset containing people:
```json
{
  "video_id": "xyz123",
  "has_identifiable_people": false,
  "consent_type": "cc_by_4.0 | commercial_license | explicit_consent_documented",
  "consent_documentation": {
    "date": "2026-02-14",
    "source": "https://example.com/license",
    "verified": true
  }
}
```

### 6.3 GDPR Right to Erasure

If a data subject requests removal:
1. **Remove from active dataset** immediately
2. **Mark in audit log** (keep for legal proof of deletion)
3. **Sync across all versions** and backups
4. **Document deletion** with requestor information (minimal, encrypted)

---

## 7. Data Lineage & Attribution

### 7.1 Training Attribution

Every trained model MUST document:

```yaml
model_name: "aiprod-v1-base"
training_dataset:
  version: "v1.0"
  total_hours: 163.1
  sources:
    - name: "Something-Something v2"
      hours: 100.2
      license: "CC-BY 4.0"
      link: "https://..."
    - name: "WebVid"
      hours: 50.1
      license: "CC-BY 4.0"
      link: "https://..."
    - name: "Custom Licensed"
      hours: 12.8
      license: "Commercial"
      link: internal
```

### 7.2 Audit Trail

```bash
# Immutable audit log
datasets/audit/
├── v1_ingestion_log.json
│   └── every download + MD5 hash recorded
├── v1_processing_log.json
│   └── scene splits, captions, validations logged
├── v1_training_log.json
│   └── which model used this data, when, with what config
└── v1_access_log.json
    └── who accessed/modified, timestamps
```

---

## 8. Policies & Compliance

### 8.1 Fair Use Doctrine (US/EU)

AIPROD's dataset usage complies with fair use precedents:
- ✅ Purpose: Training AI model (transformative)
- ✅ Amount: Reasonable sample of publicly available content
- ✅ Effect: Not cannibalizing original market
- ✅ Licenses: Preference for explicit CC-BY, commercial terms where applicable

### 8.2 Prohibited Activities

🚫 **Strictly Forbidden:**
1. Using datasets for commercial competitive training without explicit license
2. Storing/processing data outside GDPR-compliant jurisdictions
3. Sharing training data publicly without original license permission
4. Training models that directly replicate copyrighted content
5. Reverse-engineering proprietary datasets from model outputs

### 8.3 Modification of Datasets

If datasets are modified:
- ✅ Document all transformations (scene splits, captions, resampling)
- ✅ Update license attribution to include AIPROD modifications
- ✅ Preserve original license chain
- ✅ Make modifications available under same license (copyleft requirement)

---

## 9. Data Quality Audit Process

### 9.1 QC Checklist

Every 1,000 videos requires manual QC sampling:

```checklist
□ Resolution adequate for target use case
□ No watermarks/logos >10% frame
□ Audio sync within ±100ms
□ No pure black/white screens >2s
□ Frame rate consistent
□ No heavy encoding artifacts
□ Aspect ratio reasonable
□ Content matches provided captions
□ No NSFW/inappropriate content
□ License metadata correct
```

### 9.2 Automated Validation Pipeline

```python
def validate_video(video_path) -> QualityScore:
    """Automated QC scoring"""
    checks = {
        "resolution": verify_resolution(),       # 1920×1080 ≥ target
        "framerate": verify_fps(),               # 24-60 fps
        "duration": verify_duration(),           # 2-600 sec
        "audio_present": verify_audio(),         # Mono+/Stereo required
        "aspect_ratio": verify_aspect(),         # 4:3 to 21:9
        "black_frames": detect_black_frames(),   # <2% per second
        "encoding_quality": ffprobe_quality(),   # Bitrate, codec check
        "watermark": detect_watermark(),         # Area %
        "faces": detect_identifiable_faces(),    # Privacy check
    }
    
    score = weighted_average(checks)  # 0.0 → 1.0
    
    if score >= 0.85:
        return APPROVED
    elif score >= 0.70:
        return NEEDS_REVIEW
    else:
        return REJECTED
```

### 9.3 Quarterly Audit Report

Every 3 months:
- Sample 5% of dataset videos
- Re-score quality
- Document quality trends
- Identify systematic issues in specific sources
- Update QC thresholds if needed

---

## 10. Implementation Roadmap

| Phase | Timeline | Action |
|-------|----------|--------|
| **Phase 0** | Weeks 1-2 | Dataset policy finalized (THIS DOCUMENT) |
| **Phase 1a** | Weeks 3-6 | Public dataset collection (Kinetics, WebVid, etc.) |
| **Phase 1b** | Weeks 7-10 | Commercial licensing (Shutterstock, Getty) |
| **Phase 1c** | Weeks 11-16 | DVC setup + ingestion pipeline |
| **Phase 2** | Weeks 17-32 | Data at scale (100K+ videos) |
| **Phase 3** | Weeks 33-48 | Specialized dataset curation |
| **Phase 4** | Weeks 49+ | Continuous dataset maintenance & updates |

---

## 11. Roles & Responsibilities

| Role | Responsibility |
|------|-----------------|
| **Data Lead** | Dataset policy enforcement, vendor management |
| **Data Engineer** | Ingestion pipeline, DVC versioning, quality metrics |
| **ML Engineer** | Quality threshold tuning, training integration |
| **Legal/Compliance** | License verification, GDPR compliance, IP review |
| **QA Team** | Manual QC sampling, audit trails |

---

## 12. Contact & Questions

For dataset governance questions:
- **Data Policy:** data-governance@aiprod.internal
- **Technical Issues:** data-engineering@aiprod.internal
- **Compliance/Legal:** legal@aiprod.internal

---

**Approval Chain:**
- ✅ Created: February 14, 2026
- ⏳ Next Review: May 14, 2026 (Quarterly)
- ⏳ Compliance Audit: Requires external counsel sign-off

---

## Appendix A: License Quick Reference

```
CC-BY 4.0:        ✅ Commercial use OK, must attribute
CC-BY-SA 4.0:     ✅ Commercial use OK, derivative under CC-BY-SA
CC-BY-NC:         🚫 NO commercial use
CC0 (Public):     ✅ No restrictions
Apache 2.0:       ✅ No restrictions, must include boilerplate
GPL/AGPL:         ⚠️  Requires careful legal review
Commercial:       ✅ Explicit terms from vendor
```

## Appendix B: Sample DVC Commands

```bash
# Initialize DVC in project
dvc init

# Add dataset to DVC
dvc add datasets/v1/raw/

# Create remote storage
dvc remote add -d myremote s3://aiprod-datasets

# Push to S3
dvc push

# Pull specific version
dvc pull -r myremote datasets/v1.dvc

# Track changes
git add .dvc/ datasets/.gitignore
git commit -m "Dataset v1.0: Initial 163 hours curated"
```

