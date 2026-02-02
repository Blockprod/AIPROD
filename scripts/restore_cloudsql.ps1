param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [Parameter(Mandatory = $true)]
    [string]$Instance,

    [Parameter(Mandatory = $true)]
    [string]$Database,

    [Parameter(Mandatory = $true)]
    [string]$BackupFile
)

Write-Host "Starting Cloud SQL import..." -ForegroundColor Cyan
Write-Host "Project: $ProjectId"
Write-Host "Instance: $Instance"
Write-Host "Database: $Database"
Write-Host "Backup: $BackupFile" -ForegroundColor Yellow

& gcloud sql import sql $Instance $BackupFile `
    --project=$ProjectId `
    --database=$Database

if ($LASTEXITCODE -ne 0) {
    Write-Host "Import failed" -ForegroundColor Red
    exit 1
}

Write-Host "Import completed successfully" -ForegroundColor Green
