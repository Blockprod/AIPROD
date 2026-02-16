"""
Vérifie l'intégrité de tous les fichiers de poids modèles.
Compare les SHA-256 actuels avec ceux stockés dans models/CHECKSUMS.sha256.

Usage:
    # Générer les checksums initiaux (première exécution)
    python scripts/verify_model_integrity.py --generate

    # Vérifier l'intégrité (exécutions suivantes)
    python scripts/verify_model_integrity.py
"""
import argparse
import hashlib
import sys
from pathlib import Path


MODELS_DIR = Path("models")
CHECKSUMS_FILE = MODELS_DIR / "CHECKSUMS.sha256"
WEIGHT_EXTENSIONS = {".safetensors", ".pt", ".bin", ".ckpt"}


def compute_sha256(filepath: Path) -> str:
    """Calcule le SHA-256 d'un fichier."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_weight_files() -> list[Path]:
    """Trouve tous les fichiers de poids dans models/."""
    files = []
    for ext in WEIGHT_EXTENSIONS:
        files.extend(MODELS_DIR.rglob(f"*{ext}"))
    return sorted(files)


def generate_checksums() -> None:
    """Génère le fichier CHECKSUMS.sha256."""
    weight_files = find_weight_files()
    if not weight_files:
        print("⚠️  Aucun fichier de poids trouvé dans models/")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    for f in weight_files:
        rel_path = f.relative_to(MODELS_DIR)
        print(f"  Hashing {rel_path} ...", end="", flush=True)
        sha = compute_sha256(f)
        size_mb = f.stat().st_size / (1024 * 1024)
        lines.append(f"{sha}  {rel_path}")
        print(f" {sha[:16]}... ({size_mb:.1f} MB)")

    with open(CHECKSUMS_FILE, "w", encoding="utf-8") as cf:
        cf.write("\n".join(lines) + "\n")

    print(f"\n✅ {len(lines)} checksums écrits dans {CHECKSUMS_FILE}")


def verify_checksums() -> bool:
    """Vérifie l'intégrité des fichiers contre CHECKSUMS.sha256."""
    if not CHECKSUMS_FILE.exists():
        print(f"❌ {CHECKSUMS_FILE} non trouvé. Exécutez d'abord avec --generate")
        return False

    with open(CHECKSUMS_FILE, encoding="utf-8") as cf:
        lines = [line.strip() for line in cf if line.strip()]

    ok = 0
    fail = 0
    missing = 0

    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            continue
        expected_sha, rel_path = parts
        filepath = MODELS_DIR / rel_path

        if not filepath.exists():
            print(f"  ❌ MANQUANT:  {rel_path}")
            missing += 1
            continue

        actual_sha = compute_sha256(filepath)
        if actual_sha == expected_sha:
            print(f"  ✅ OK:        {rel_path}")
            ok += 1
        else:
            print(f"  ❌ CORROMPU:  {rel_path}")
            print(f"     attendu: {expected_sha}")
            print(f"     obtenu:  {actual_sha}")
            fail += 1

    print(f"\nRésultat: {ok} OK, {fail} corrompus, {missing} manquants")
    return fail == 0 and missing == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérification d'intégrité des modèles AIPROD")
    parser.add_argument(
        "--generate", action="store_true",
        help="Générer les checksums (première exécution ou après MAJ des modèles)"
    )
    args = parser.parse_args()

    if args.generate:
        generate_checksums()
    else:
        success = verify_checksums()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
