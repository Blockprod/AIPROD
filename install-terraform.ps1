# Script d'installation de Terraform
$ErrorActionPreference = "Stop"

Write-Host "⬇️  Installation de Terraform 1.7.0..." -ForegroundColor Cyan

$terraformVersion = "1.7.0"
$downloadUrl = "https://releases.hashicorp.com/terraform/${terraformVersion}/terraform_${terraformVersion}_windows_amd64.zip"
$zipPath = "$env:TEMP\terraform_${terraformVersion}.zip"
$extractPath = "C:\Program Files\Terraform"

try {
    # Créer le dossier
    Write-Host "📁 Création du dossier: $extractPath" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $extractPath -Force | Out-Null
    
    # Télécharger
    Write-Host "📥 Téléchargement depuis: $downloadUrl" -ForegroundColor Yellow
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -TimeoutSec 120
    Write-Host "✅ Téléchargement complété" -ForegroundColor Green
    
    # Extraire
    Write-Host "📦 Extraction vers: $extractPath" -ForegroundColor Yellow
    Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
    Write-Host "✅ Extraction complétée" -ForegroundColor Green
    
    # Nettoyer
    Remove-Item $zipPath -Force
    
    # Vérifier
    $tfPath = "$extractPath\terraform.exe"
    if (Test-Path $tfPath) {
        Write-Host "✅ terraform.exe installé avec succès: $tfPath" -ForegroundColor Green
        Write-Host ""
        Write-Host "Version:" -ForegroundColor Cyan
        & $tfPath --version
        
        Write-Host ""
        Write-Host "📍 Ajout de Terraform au PATH de la session..." -ForegroundColor Yellow
        $env:PATH = "$extractPath;$env:PATH"
        Write-Host "✅ PATH mis à jour" -ForegroundColor Green
    }
    else {
        Write-Host "❌ terraform.exe non trouvé après extraction" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "❌ Erreur: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 Installation complétée avec succès !" -ForegroundColor Green
