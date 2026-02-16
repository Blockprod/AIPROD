# AIPROD Sovereign Model Card

## Résumé

Collection de modèles 100% propriétaires pour la génération vidéo IA.
Tous les modèles sont entraînés sur des données sous licence et fonctionnent **entièrement hors-ligne**, sans aucune dépendance à des API cloud tierces.

## Modèles

### 1. AIPROD-SHDT v1 (Transformer de Diffusion)

| Propriété | Valeur |
|---|---|
| **Architecture** | SHDT (Sovereign Hybrid Diffusion Transformer) |
| **Paramètres** | 19B |
| **Format** | FP8 E4M3 (quantifié via optimum-quanto) |
| **Taille** | ~12 GB |
| **Résolution max** | 1024×576 (16:9) à 97 frames |
| **Entraînement** | Full fine-tuning 4 phases avec curriculum progressif |
| **Config** | `configs/train/full_finetune.yaml` |

**Pipeline d'entraînement :**
1. Phase 1 — LoRA rank=32, 256×256, 16 frames, 15k steps (`configs/train/lora_phase1.yaml`)
2. Phase 2 — Full fine-tune avec curriculum progressif (256→512→768→1024), 100k steps
3. Phase 3 — Quantization FP8 via `scripts/quantize_model.py`

### 2. AIPROD-HWVAE v1 (Video VAE)

| Propriété | Valeur |
|---|---|
| **Architecture** | HW-VAE (Haar Wavelet Video Autoencoder) |
| **Paramètres** | ~150M |
| **Format** | bfloat16 |
| **Taille** | ~500 MB |
| **Loss** | L1 + Perceptual (VGG16) + Spectral + KL |
| **Config** | `configs/train/vae_finetune.yaml` |

**Caractéristiques :** Décomposition Haar Wavelet multi-échelle, compression spatiale ×8, compression temporelle ×4.

### 3. AIPROD-Audio-VAE v1 (Audio Codec)

| Propriété | Valeur |
|---|---|
| **Architecture** | NAC (Neural Audio Codec) + RVQ |
| **Paramètres** | ~50M |
| **Format** | bfloat16 |
| **Taille** | ~200 MB |
| **Sample rate** | 24 kHz |
| **Loss** | Reconstruction + RVQ commitment + Spectral + Adversarial |
| **Config** | `configs/train/audio_vae.yaml` |

**Caractéristiques :** Encodeur causal 1D, Residual Vector Quantization à 8 codebooks de 1024 entrées, décodeur avec upsampling progressif.

### 4. AIPROD-TTS v1

| Propriété | Valeur |
|---|---|
| **Architecture** | FastSpeech 2 + HiFi-GAN + ProsodyModeler |
| **Paramètres** | ~80M |
| **Format** | bfloat16 |
| **Taille** | ~300 MB |
| **Config** | `configs/train/tts_training.yaml` |

**Pipeline TTS :**
- **TextFrontend** : Tokenization + phonème embedding
- **MelDecoder** : Prédiction mel-spectrogram (FastSpeech 2)
- **VocoderTTS** : Synthèse waveform (HiFi-GAN)
- **ProsodyModeler** : Contrôle pitch, énergie, durée

**Données d'entraînement :**
- LJSpeech (13k clips, 24h, domaine public)
- LibriTTS (585h, CC BY 4.0)

### 5. AIPROD-Text-Encoder v1

| Propriété | Valeur |
|---|---|
| **Architecture** | Gemma-3-1B + LoRA projection |
| **Paramètres** | ~1B |
| **Format** | bfloat16 |
| **Taille** | ~1 GB |
| **Base** | Gemma-3-1B (Apache 2.0, Google) |

**Adaptation :** Fine-tuning LoRA (rank=32) sur 100k paires (prompt, video caption) pour projeter les embeddings texte vers l'espace latent SHDT.

### 6. AIPROD-Upsampler v1

| Propriété | Valeur |
|---|---|
| **Architecture** | Spatial Upsampler ×2 |
| **Paramètres** | ~150M |
| **Format** | bfloat16 |
| **Taille** | ~500 MB |

## Souveraineté

| Critère | Statut |
|---|---|
| Dépendances cloud | **0** |
| APIs tierces requises | **0** |
| Poids stockés localement | **100%** |
| Fonctionnement hors-ligne | **✅ Complet** |
| Licence des données d'entraînement | **Vérifiée** (domaine public / CC BY / propriétaire) |

## Checksums

Tous les fichiers `.safetensors` disposent d'un SHA-256 calculé et enregistré dans `MANIFEST.json`.
Vérification :
```bash
sha256sum -c CHECKSUMS.sha256
```

## Licence

Tous les poids de modèles dans ce répertoire sont sous **licence propriétaire AIPROD**, à l'exception de :
- `aiprod-text-encoder-v1` : poids de base Gemma-3-1B sous Apache 2.0, poids LoRA propriétaires.

## Contact

AIPROD — Modèles vidéo IA souverains.
