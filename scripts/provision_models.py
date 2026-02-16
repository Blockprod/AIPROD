"""
Télécharge et stocke TOUS les modèles nécessaires en local.
À exécuter UNE FOIS sur machine connectée, puis le projet est 100% offline.

Usage:
    python scripts/provision_models.py

Après exécution, tous les modèles sont dans models/ et un MANIFEST.json est créé.
Le projet peut ensuite fonctionner en mode air-gapped (AIPROD_OFFLINE=1).
"""
from huggingface_hub import snapshot_download
from pathlib import Path
import hashlib
import json


MODELS = {
    # ── Training initialization bases (needed ONLY for fine-tuning on Colab) ──
    # After training, these are NO LONGER needed. Inference uses proprietary models.
    "models/training-bases/gemma-3-1b": {
        "repo": "google/gemma-3-1b-pt",
        "revision": "main",  # Figer au commit SHA exact après téléchargement
        "purpose": "training_base_only",
    },
    "models/scenarist/mistral-7b": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "main",
    },
    "models/captioning/qwen-omni-7b": {
        "repo": "Qwen/Qwen2.5-Omni-7B",
        "revision": "main",
    },
    "models/clip/clip-vit-large-patch14": {
        "repo": "openai/clip-vit-large-patch14",
        "revision": "main",
    },
}


def compute_sha256(filepath: Path) -> str:
    """Calcule le SHA-256 d'un fichier."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_all():
    """Télécharge tous les modèles et génère le manifeste."""
    manifest = {}
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    for local_path, spec in MODELS.items():
        path = Path(local_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"⬇️  Downloading {spec['repo']} → {local_path}")

        snapshot_download(
            repo_id=spec["repo"],
            local_dir=str(path),
            revision=spec["revision"],
        )

        # Collecter les fichiers de poids avec leurs checksums
        weight_files = {}
        for ext in ("*.safetensors", "*.pt", "*.bin"):
            for f in path.rglob(ext):
                rel = str(f.relative_to(path))
                weight_files[rel] = compute_sha256(f)

        manifest[local_path] = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "files": weight_files,
        }
        print(f"   ✅ {len(weight_files)} fichiers de poids trouvés")

    # Sauvegarder le manifeste
    manifest_path = models_dir / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Tous les modèles provisionnés. {manifest_path} créé.")
    print(f"   {len(manifest)} modèles, {sum(len(v['files']) for v in manifest.values())} fichiers totaux")

    # Pré-provisionner VGG16 dans le cache PyTorch local
    print("\n⬇️  Pré-provisionnement VGG16 (perte perceptuelle VAE)...")
    try:
        from torchvision.models import vgg16, VGG16_Weights
        vgg16(weights=VGG16_Weights.DEFAULT)
        print("   ✅ VGG16 dans le cache PyTorch local")
    except Exception as e:
        print(f"   ⚠️ VGG16 non provisionné: {e}")

    print("\n🔒 Projet prêt pour le mode air-gapped (AIPROD_OFFLINE=1)")


if __name__ == "__main__":
    download_all()
