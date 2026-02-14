# Script pour télécharger les modèles LTX-2 pour research
# ⚠️ USAGE: Étude seulement, pas utilisation commerciale sans respecter Apache 2.0

Write-Host "=== Téléchargement LTX-2 Research Models ===" -ForegroundColor Cyan
Write-Host ""

# Variables
$HF_CLI = "huggingface-cli"
$DOWNLOAD_DIR = ".\models\ltx2_research"
$REPO = "Lightricks/LTX-2"

# Vérifier que huggingface-cli est installé
$hf_installed = Get-Command $HF_CLI -ErrorAction SilentlyContinue
if (-not $hf_installed) {
    Write-Host "❌ huggingface-cli non trouvé" -ForegroundColor Red
    Write-Host "Installez-le: pip install huggingface-hub" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ huggingface-cli détecté" -ForegroundColor Green
Write-Host ""

# Créer le répertoire s'il n'existe pas
if (-not (Test-Path $DOWNLOAD_DIR)) {
    New-Item -ItemType Directory -Path $DOWNLOAD_DIR -Force | Out-Null
    Write-Host "📁 Créé: $DOWNLOAD_DIR" -ForegroundColor Green
}

Write-Host ""
Write-Host "Sélectionnez les modèles à télécharger:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. [RECOMMANDÉ] FP8 (18GB) + Upscaler (optimal pour GTX 1070)"
Write-Host "2. Dev full precision (40GB + upscaler, plus lent)"
Write-Host "3. Distilled (10GB, très rapide mais moins exact)"
Write-Host ""

$choice = Read-Host "Votre choix (1-3)"

Write-Host ""

switch ($choice) {
    "1" {
        Write-Host "📥 Téléchargement: ltx-2-19b-dev-fp8.safetensors (18GB)" -ForegroundColor Cyan
        & $HF_CLI download $REPO `
            --repo-type model `
            --local-dir $DOWNLOAD_DIR `
            --include "ltx-2-19b-dev-fp8.safetensors"

        Write-Host ""
        Write-Host "📥 Téléchargement: ltx-2-spatial-upscaler-x2-1.0.safetensors (6GB)" -ForegroundColor Cyan
        & $HF_CLI download $REPO `
            --repo-type model `
            --local-dir $DOWNLOAD_DIR `
            --include "ltx-2-spatial-upscaler-x2-1.0.safetensors"
        
        Write-Host ""
        Write-Host "✅ Téléchargement RECOMMANDÉ terminé (~24GB)" -ForegroundColor Green
    }
    "2" {
        Write-Host "📥 Téléchargement: ltx-2-19b-dev.safetensors (40GB)" -ForegroundColor Cyan
        & $HF_CLI download $REPO `
            --repo-type model `
            --local-dir $DOWNLOAD_DIR `
            --include "ltx-2-19b-dev.safetensors"

        Write-Host ""
        Write-Host "📥 Téléchargement: ltx-2-spatial-upscaler-x2-1.0.safetensors (6GB)" -ForegroundColor Cyan
        & $HF_CLI download $REPO `
            --repo-type model `
            --local-dir $DOWNLOAD_DIR `
            --include "ltx-2-spatial-upscaler-x2-1.0.safetensors"
        
        Write-Host ""
        Write-Host "✅ Téléchargement complet (~46GB total)" -ForegroundColor Green
    }
    "3" {
        Write-Host "📥 Téléchargement: ltx-2-19b-distilled-fp8.safetensors (5GB)" -ForegroundColor Cyan
        & $HF_CLI download $REPO `
            --repo-type model `
            --local-dir $DOWNLOAD_DIR `
            --include "ltx-2-19b-distilled-fp8.safetensors"

        Write-Host ""
        Write-Host "✅ Téléchargement rapide terminé (~5GB)" -ForegroundColor Green
    }
    default {
        Write-Host "❌ Choix invalide" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "📊 Fichiers téléchargés:" -ForegroundColor Cyan
Get-ChildItem $DOWNLOAD_DIR -File | ForEach-Object {
    $size = [math]::Round($_.Length / 1GB, 2)
    Write-Host "  - $($_.Name): $size GB"
}

Write-Host ""
Write-Host "⚠️  IMPORTANT: Ces modèles sont pour ÉTUDE seulement (Apache 2.0)" -ForegroundColor Yellow
Write-Host "   Consultez models/ltx2_research/README_RESEARCH.md" -ForegroundColor Yellow
Write-Host ""
Write-Host "✅ Prêt pour Phase 0: Research & Learning" -ForegroundColor Green
Write-Host ""
Write-Host "Prochaines étapes:" -ForegroundColor Cyan
Write-Host "1. Lire les modèles et prendre des notes"
Write-Host "2. Documenter insights dans AIPROD_V2_RESEARCH_NOTES.md"
Write-Host "3. Concevoir AIPROD architecture NOUVELLE (pas copie)"
Write-Host "4. Phase 1: Créer backbone AIPROD original" -ForegroundColor Cyan
