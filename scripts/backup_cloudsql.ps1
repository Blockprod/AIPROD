param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [Parameter(Mandatory = $true)]
    [string]$Instance,

    [Parameter(Mandatory = $true)]
    [string]$Database,

    [Parameter(Mandatory = $true)]
    [string]$Bucket,

    [string]$BackupPrefix = "aiprod-v33-backup"
)

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupFile = "$BackupPrefix-$Database-$timestamp.sql.gz"
$gcsPath = "gs://$Bucket/$backupFile"

Write-Host "Starting Cloud SQL export..." -ForegroundColor Cyan
Write-Host "Project: $ProjectId"
Write-Host "Instance: $Instance"
Write-Host "Database: $Database"
Write-Host "Destination: $gcsPath" -ForegroundColor Yellow

& gcloud sql export sql $Instance $gcsPath `
    --project=$ProjectId `
    --database=$Database `
    --offload

if ($LASTEXITCODE -ne 0) {
    Write-Host "Export failed" -ForegroundColor Red
    exit 1
}

Write-Host "Export completed: $gcsPath" -ForegroundColor Green
