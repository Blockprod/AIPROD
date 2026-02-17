#!/usr/bin/env python
"""
Diagnostic et réparation du dossier sovereign/
"""

import os
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime

# === Configuration ===
SOURCE_LOCATION = Path('models') / 'aiprod-sovereign'
SOVEREIGN_DIR = Path('sovereign')

print("=" * 70)
print("🔧 DIAGNOSTIC - Vérification sovereign/")
print("=" * 70)
print()

# === Vérifier la source ===
if not SOURCE_LOCATION.exists():
    print(f"❌ Dossier source {SOURCE_LOCATION} introuvable")
    exit(1)

print(f"✅ Source trouvée: {SOURCE_LOCATION}")
print()

# === Lister ce qui existe ===
print("📋 Fichiers dans models/aiprod-sovereign/:")
print()

existing_files = {}
for f in sorted(SOURCE_LOCATION.rglob('*')):
    if f.is_file() and f.suffix in ['.safetensors', '.json']:
        rel_path = f.relative_to(SOURCE_LOCATION)
        size_gb = f.stat().st_size / 1024**3
        print(f"   ✅ {rel_path}: {size_gb:.2f} GB")
        existing_files[str(rel_path)] = f
        
if not existing_files:
    print("   ❌ AUCUN FICHIER TROUVÉ!")
    exit(1)

print()

# === Créer/Nettoyer sovereign/ ===
print("📦 Préparation du dossier sovereign/...")
SOVEREIGN_DIR.mkdir(parents=True, exist_ok=True)

# Copier les fichiers .safetensors
print()
for rel_path_str, src_file in existing_files.items():
    if src_file.suffix == '.safetensors':
        dst = SOVEREIGN_DIR / src_file.name
        shutil.copy2(src_file, dst)
        size_gb = dst.stat().st_size / 1024**3
        print(f"✅ {src_file.name} ({size_gb:.2f} GB)")

# Copier le text encoder (si c'est un dossier)
te_src = SOURCE_LOCATION / 'aiprod-text-encoder-v1'
if te_src.exists() and te_src.is_dir():
    te_dst = SOVEREIGN_DIR / 'aiprod-text-encoder-v1'
    if te_dst.exists():
        shutil.rmtree(te_dst)
    shutil.copytree(te_src, te_dst)
    print(f"✅ aiprod-text-encoder-v1/ (dossier)")

print()
print("=" * 70)
print("⚠️  FICHIERS MANQUANTS")
print("=" * 70)
print()

expected = {
    'aiprod-shdt-v1-fp8.safetensors': 'Transformer de diffusion vidéo',
    'aiprod-hwvae-v1.safetensors': 'Video VAE (HW)',
    'aiprod-audio-vae-v1.safetensors': 'Audio codec (NAC)',
    'aiprod-tts-v1.safetensors': 'Text-to-Speech',
}

missing = []
for fname, description in expected.items():
    if (SOVEREIGN_DIR / fname).exists():
        print(f"✅ {fname}")
    else:
        print(f"❌ {fname} — {description}")
        missing.append(fname)

print()
print("=" * 70)
print("📊 RÉSUMÉ")
print("=" * 70)
print()

files_in_sovereign = len(list(SOVEREIGN_DIR.rglob('*')))
total_size = sum(f.stat().st_size for f in SOVEREIGN_DIR.rglob('*') if f.is_file()) / 1024**3

print(f"📁 sovereign/ contient: {files_in_sovereign} fichiers")
print(f"💾 Taille totale: {total_size:.2f} GB")
print()

if missing:
    print(f"⚠️  {len(missing)} fichier(s) manquant(s) :")
    for fname in missing:
        print(f"   - {fname}")
    print()
    print("RAISON PROBABLE: Les phases D2, D3, D4 n'ont PAS TERMINÉ sur Colab")
    print()
    print("SOLUTIONS:")
    print("  1. RÉ-LANCER les cellules D2, D3, D4 sur Colab")
    print("  2. Augmenter D1a avec 50k+ steps LoRA (au lieu de 15k)")
    print("  3. Utiliser un cloud VM avec 4× A100-80GB pour D1b")
else:
    print("✅ TOUS LES FICHIERS SONT PRÉSENTS!")

print()

# === Générer MANIFEST.json ===
print("📋 Génération du MANIFEST.json...")
print()

manifest = {
    "version": "1.0.0",
    "name": "aiprod-sovereign",
    "description": "Modèles propriétaires AIPROD",
    "exported_date": datetime.now().isoformat(),
    "status": "INCOMPLETE" if missing else "COMPLETE",
    "missing_components": missing,
    "sovereignty": {
        "score": f"{(4 - len(missing))/4 * 10:.0f}/10",
        "proprietary_weights": True,
        "external_dependencies": 0 if not missing else len(missing),
        "offline_capable": not bool(missing),
    },
    "models": {}
}

for f in sorted(SOVEREIGN_DIR.rglob('*')):
    if f.is_file() and f.suffix in ['.safetensors', '.json']:
        try:
            sha = hashlib.sha256(f.read_bytes()).hexdigest()
            rel_path = str(f.relative_to(SOVEREIGN_DIR))
            size_gb = round(f.stat().st_size / 1024**3, 2)
            manifest['models'][rel_path] = {
                'sha256': sha,
                'size_bytes': f.stat().st_size,
                'size_gb': size_gb,
                'status': 'trained',
                'license': 'Propriétaire AIPROD'
            }
            print(f"   {rel_path}: SHA={sha[:16]}...")
        except Exception as e:
            print(f"   ⚠️ {f.name}: {e}")

manifest_path = SOVEREIGN_DIR / 'MANIFEST.json'
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

print()
print("✅ MANIFEST.json créé")
print()
