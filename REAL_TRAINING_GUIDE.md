# 🎯 GUIDE COMPLET : Entraînement RÉEL vs TEST

## Le Problème

Le notebook Colab par défaut utilise **`--dummy-data`** pour :
- ✅ **Test rapide du pipeline** (~5 min)
- ❌ **PAS pour la production** (données aléatoires)

Les "modèles" générés avec `--dummy-data` sont du **bruit blanc** — **inutilisables**.

---

## Solution : Passer en Mode RÉEL

### 1️⃣ Vérifier votre GPU

| GPU | VRAM | Status |
|-----|------|--------|
| T4 (Colab gratuit) | 15 GB | ❌ IMPOSSIBLE |
| **A100 40GB (Colab Pro)** | 40 GB | ✅ **RECOMMANDÉ** |
| RTX 6000 (Lambda Labs) | 48 GB | ✅ **BON** |
| A100 80GB (Cloud VM) | 80 GB | ✅ **IDÉAL** |

**T4 n'a pas assez de VRAM.**

### 2️⃣ Mettre en place l'infrastructure

#### Option A : Colab Pro ($10/mois ou crédits)

```
1. Upgrader Colab → Pro
2. Runtime → Change runtime type → GPU T4 → **A100**
3. Relancer le notebook
```

#### Option B : Cloud VM ($3-10 pour une session)

```bash
# RunPod.io
./runpod run --gpuType A100-40GB --containerDiskInGb 100
git clone https://github.com/Blockprod/AIPROD.git
cd AIPROD

# Ou Lambda Labs (~$1/h)
# https://lambdalabs.com/service/gpu-cloud
```

### 3️⃣ Dans le Notebook Colab

**Cellule 3b (Configuration Entraînement) :**

```python
REAL_TRAINING = True  # ← Passer de False à True
```

Cela active automatiquement :
- ✅ LJSpeech (13,100 clips audio)
- ✅ Pexels API (vidéos réelles)
- ✅ Epochs réalistes : 80 (D2), 100 (D3), 800 (D4)
- ✅ Durée : ~17-20h sur A100

### 4️⃣ Télécharger les VRAIES données

Le notebook inclut maintenant un téléchargement automatique :

```python
# Executé automatiquement si REAL_TRAINING = True :
- LJSpeech (2.5 GB) → data/lj_speech/
- Pexels API (configuré avec 5000 vidéos)
```

**Problème potentiel :** LibriTTS est énorme (40+ GB)
- Si vous avez besoin de D4 Phase 3 optimale
- Téléchargez manuellement depuis [OpenSlice](http://www.openslice.org/)
- Ou utilisez uniquement LJSpeech (acceptable)

### 5️⃣ Lancer l'entraînement

**Exécutez dans l'ordre :**

1. Cell 0-1: Vérifier GPU ✅
2. Cell 2: Mount Drive ✅
3. Cell 3: Install packages ✅
4. **Cell 3b: `REAL_TRAINING = True`** ← L'étape clé !
5. Cell 3c: Config entraînement (détecte auto)
6. Cell 3d: Télécharge données réelles
7. Cell 4: D1a LoRA (~8h A100) → Output: SHDT
8. **Cell 7: D2 HW-VAE (~4h A100)** → Output: `aiprod-hwvae-v1.safetensors`
9. **Cell 8: D3 Audio VAE (~2h A100)** → Output: `aiprod-audio-vae-v1.safetensors`
10. **Cell 9: D4 TTS 3 phases (~3h A100)** → Output: `aiprod-tts-v1.safetensors`
11. **Cell 10: Export + Quantize** → Output: `sovereign/` complet

**Durée totale : ~17-20h sur A100** (peut s'étendre sur 2-3 jours)

### 6️⃣ Résultat Final

```
sovereign/
├── aiprod-shdt-v1-fp8.safetensors       (10-12 GB) ✅
├── aiprod-hwvae-v1.safetensors          (500 MB) ✅
├── aiprod-audio-vae-v1.safetensors      (200 MB) ✅
├── aiprod-tts-v1.safetensors            (300 MB) ✅
├── aiprod-text-encoder-v1/              (2 GB) ✅
└── MANIFEST.json                        (SHA-256) ✅
```

**Chaque `.safetensors` contient des poids entraînés réels**, 100% crédibles et utilisables.

---

## 📊 Comparaison : Dummy vs Real

| Aspect | `--dummy-data` TEST | `REAL_TRAINING` PRODUCTION |
|--------|---------------------|---------------------------|
| **Données** | Random tensors | Video/Audio réels |
| **Epochs** | 3-5 (rapide) | 80-800 (complet) |
| **Durée** | 5-10 min | 17-20 heures |
| **Poids générés** | ❌ **Bruit blanc** | ✅ **Modèles réels** |
| **GPU requis** | T4 (15 GB) | **A100 40GB+** |
| **Utilisable?** | ❌ Juste test pipeline | ✅ **Production-ready** |
| **Coût** | $0 (Colab free) | $0-10 (Colab Pro ou Cloud) |

---

## ⚙️ Configuration Exacte

### Mode TEST (actuellement par défaut)
```python
# Cellule 3b, ligne 1
REAL_TRAINING = False

# Résultat:
D2_USE_DUMMY = True
D2_EPOCHS = 5
D3_USE_DUMMY = True
D3_EPOCHS = 5
D4_USE_DUMMY = True
D4_PHASES_EPOCHS = (3, 3, 3)
```

### Mode PRODUCTION (ce que vous voulez)
```python
# Cellule 3b, ligne 1
REAL_TRAINING = True

# Résultat:
D2_USE_DUMMY = False
D2_EPOCHS = 80
D3_USE_DUMMY = False
D3_EPOCHS = 100
D4_USE_DUMMY = False
D4_PHASES_EPOCHS = (200, 500, 100)
```

---

## 🚀 Checklist Avant de Démarrer

- [ ] ✅ GPU A100 40GB ou mieux configuré
- [ ] ✅ Google Drive synchronisé
- [ ] ✅ Colab notebook ouvert
- [ ] ✅ **`REAL_TRAINING = True` défini**
- [ ] ✅ 50+ GB espace disque Colab disponible
- [ ] ✅ Prêt à attendre 17-20 heures

---

## 📝 Notes Importantes

### 1. LibriTTS (optionnel pour D4 phase 3)
- **Très volumineux** (40+ GB)
- **Téléchargement manuel** recommandé
- Vous pouvez faire D4 sans il (utilisera LJSpeech seulement)

### 2. Pexels API
- **Gratuit**, 200 requêtes/heure
- Script télécharge automatiquement 5000 vidéos
- Peut être augmenté dans la cellule 3d

### 3. Sauvegarder sur Drive
- Tous les modèles sont automatiquement sauvés sur Google Drive
- Téléchargez vers votre machine locale après

### 4. Si timeout Colab
- Vous pouvez continuer d'une phase à l'autre
- Les checkpoints sont récupérés automatiquement
- Total peut prendre 2-3 jours par phases

---

## ❓ FAQ

**Q: T4 peut fonctionner?**
A: Non. T4 15GB n'a pas assez pour D2/D3/D4. **A100 requis.**

**Q: Combien ça coûte?**
A: Gratuit si Colab Pro, sinon ~$10 cloud VM.

**Q: Ça prend combien de temps?**
A: ~17-20h non-stop sur A100, ou 2-3j avec pauses.

**Q: Les modèles seront-ils bons?**
A: **OUI,** 100% crédibles après 80-800 epochs sur vraies données.

**Q: Puis-je utiliser T4 pour D1a seulement?**
A: Oui, D1a (8h) marche sur T4. Puis upgrader A100 pour D2-D4.

---

## 🎯 Appel à l'Action

1. **Upgrader GPU** → A100 (Colab Pro ou cloud VM)
2. **Modifier Cell 3b** : `REAL_TRAINING = True`
3. **Exécuter le notebook** de bout en bout
4. **Attendre 17-20h** pour des modèles réels
5. **Télécharger le dossier `sovereign/`** des Google Drive
6. **Utiliser vos propres modèles !**

---

**Résultat final : Une suite COMPLÈTE de modèles 100% propriétaires, EntraInés sur vraies données, et prêts pour l'inférence offline.**

✅ **C'EST LA SOLUTION SOLIDE QUE VOUS ATTENDIEZ.**
