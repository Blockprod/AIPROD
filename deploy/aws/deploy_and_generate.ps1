<#
.SYNOPSIS
    AIPROD — Déploiement et génération vidéo sur AWS EC2 GPU

.DESCRIPTION
    Script PowerShell qui automatise :
    1. Upload du code et modèles vers l'instance EC2
    2. Installation des dépendances  
    3. Génération de vidéo
    4. Téléchargement du résultat

.PARAMETER Ip
    Adresse IP publique de l'instance EC2

.PARAMETER KeyFile
    Chemin vers le fichier .pem de la clé SSH

.PARAMETER Prompt
    Prompt texte pour la génération vidéo

.PARAMETER Action
    Action: 'upload', 'generate', 'download', 'all'

.EXAMPLE
    .\deploy\aws\deploy_and_generate.ps1 -Ip 54.123.45.67 -KeyFile ~\.ssh\aiprod-key.pem -Prompt "A drone over mountains" -Action all
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Ip,
    
    [Parameter(Mandatory=$true)]
    [string]$KeyFile,
    
    [string]$Prompt = "A drone flies slowly over a mountain lake at golden hour, cinematic quality",
    
    [ValidateSet('upload', 'setup', 'generate', 'download', 'all')]
    [string]$Action = 'all',

    [int]$Height = 512,
    [int]$Width = 768,
    [int]$NumFrames = 61,
    [int]$Steps = 40,
    [int]$Seed = 10
)

$ErrorActionPreference = "Stop"
$SshUser = "ubuntu"
$RemoteDir = "~/aiprod"
$LocalRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$SshOpts = @("-i", $KeyFile, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10")

function Write-Header($msg) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Invoke-RemoteCommand($cmd) {
    Write-Host "  > $cmd" -ForegroundColor DarkGray
    ssh @SshOpts "${SshUser}@${Ip}" $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Remote command failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD — Send code + models to EC2
# ═══════════════════════════════════════════════════════════════════════════════
function Step-Upload {
    Write-Header "UPLOAD — Syncing code to EC2"
    
    # Create remote directories
    Invoke-RemoteCommand "mkdir -p $RemoteDir/models/ltx2_research $RemoteDir/models/aiprod-sovereign/aiprod-text-encoder-v1"
    
    # Upload code (exclude heavy dirs)
    Write-Host "  Uploading AIPROD code (~50 MB)..." -ForegroundColor Yellow
    $exclude = @(
        "--exclude", "models/",
        "--exclude", ".venv*/",
        "--exclude", "__pycache__/",
        "--exclude", "*.pyc",
        "--exclude", ".git/",
        "--exclude", "checkpoints/",
        "--exclude", "logs/",
        "--exclude", "output/",
        "--exclude", "*.pt"
    )
    rsync -avz --progress @exclude @SshOpts "${LocalRoot}/" "${SshUser}@${Ip}:${RemoteDir}/"
    
    # Upload text encoder (~1.9 GB)
    Write-Host ""
    Write-Host "  Uploading text encoder (~1.9 GB)..." -ForegroundColor Yellow
    rsync -avz --progress @SshOpts `
        "${LocalRoot}/models/aiprod-sovereign/aiprod-text-encoder-v1/" `
        "${SshUser}@${Ip}:${RemoteDir}/models/aiprod-sovereign/aiprod-text-encoder-v1/"
    
    # Upload main checkpoint (~25.2 GB) - THIS IS THE BIG ONE
    $ckpt = "${LocalRoot}\models\ltx2_research\ltx-2-19b-dev-fp8.safetensors"
    if (Test-Path $ckpt) {
        $sizeGB = [math]::Round((Get-Item $ckpt).Length / 1GB, 1)
        Write-Host ""
        Write-Host "  Uploading main checkpoint ($sizeGB GB) — this takes 20-60 min..." -ForegroundColor Yellow
        Write-Host "  TIP: Use a wired connection for faster upload" -ForegroundColor DarkYellow
        rsync -avz --progress @SshOpts `
            "${LocalRoot}/models/ltx2_research/ltx-2-19b-dev-fp8.safetensors" `
            "${SshUser}@${Ip}:${RemoteDir}/models/ltx2_research/"
    } else {
        Write-Host "  WARNING: Checkpoint not found at $ckpt" -ForegroundColor Red
        Write-Host "  You'll need to upload it manually or download it on the instance" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "  Upload complete" -ForegroundColor Green
}

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP — Install dependencies on EC2
# ═══════════════════════════════════════════════════════════════════════════════
function Step-Setup {
    Write-Header "SETUP — Installing dependencies on EC2"
    
    Invoke-RemoteCommand "cd $RemoteDir && pip install -e packages/aiprod-core -e packages/aiprod-pipelines --quiet 2>&1 | tail -3"
    Invoke-RemoteCommand "cd $RemoteDir && pip install av einops safetensors transformers accelerate peft --quiet 2>&1 | tail -3"
    
    # Run validation
    Write-Host "  Running pipeline validation..." -ForegroundColor Yellow
    Invoke-RemoteCommand "cd $RemoteDir && python3 scripts/validate_pipeline_e2e.py --device cuda"
    
    Write-Host "  Setup complete" -ForegroundColor Green
}

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE — Run video generation
# ═══════════════════════════════════════════════════════════════════════════════
function Step-Generate {
    Write-Header "GENERATE — Creating video on EC2 GPU"
    Write-Host "  Prompt: $Prompt" -ForegroundColor White
    Write-Host "  Config: ${Height}x${Width}, ${NumFrames}f, ${Steps} steps, seed=$Seed" -ForegroundColor White
    Write-Host ""
    
    $escapedPrompt = $Prompt -replace "'", "'\''"
    $cmd = "cd $RemoteDir && python3 scripts/generate_video_aws.py --prompt '$escapedPrompt' --height $Height --width $Width --num-frames $NumFrames --steps $Steps --seed $Seed --enable-fp8"
    
    Invoke-RemoteCommand $cmd
    
    Write-Host "  Generation complete" -ForegroundColor Green
}

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD — Get the video back to local machine
# ═══════════════════════════════════════════════════════════════════════════════
function Step-Download {
    Write-Header "DOWNLOAD — Fetching generated videos"
    
    $localOutput = Join-Path $LocalRoot "output"
    if (-not (Test-Path $localOutput)) {
        New-Item -ItemType Directory -Path $localOutput | Out-Null
    }
    
    # Download all mp4 files from output/
    rsync -avz --progress @SshOpts "${SshUser}@${Ip}:${RemoteDir}/output/*.mp4" "${localOutput}/"
    
    Write-Host ""
    Write-Host "  Videos downloaded to: $localOutput" -ForegroundColor Green
    Get-ChildItem "$localOutput\*.mp4" | ForEach-Object {
        Write-Host "    $($_.Name) ($([math]::Round($_.Length / 1MB, 1)) MB)" -ForegroundColor White
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

Write-Header "AIPROD — AWS GPU Video Generation"
Write-Host "  Instance : ${SshUser}@${Ip}"
Write-Host "  Action   : $Action"
Write-Host "  Prompt   : $Prompt"

# Test SSH connection
Write-Host ""
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
ssh @SshOpts "${SshUser}@${Ip}" "echo 'SSH OK'"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Cannot SSH to ${SshUser}@${Ip}" -ForegroundColor Red
    Write-Host "Check: 1) Instance is running  2) Security group allows SSH  3) Key file is correct" -ForegroundColor Red
    exit 1
}
Write-Host "  SSH connection OK" -ForegroundColor Green

switch ($Action) {
    'upload'   { Step-Upload }
    'setup'    { Step-Setup }
    'generate' { Step-Generate }
    'download' { Step-Download }
    'all'      { Step-Upload; Step-Setup; Step-Generate; Step-Download }
}

Write-Header "DONE"
Write-Host "  Cost estimate: check AWS Cost Explorer for exact charges"
Write-Host "  IMPORTANT: Stop/terminate the EC2 instance when done to avoid charges!"
Write-Host "    aws ec2 stop-instances --instance-ids <instance-id>"
Write-Host ""
