# Simple checker - lancez cette commande toutes les 1-2 minutes
Write-Host ""
Write-Host "═════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "VÉRIFICATION DU STATUT TERRAFORM" -ForegroundColor Cyan
Write-Host "═════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
Write-Host ""

# Test 1: Cloud Run URL
Write-Host "1️⃣  CLOUD RUN URL:" -ForegroundColor Yellow
$url = terraform output -raw cloud_run_url 2>&1
if ($url -like "https://*") {
    Write-Host "   ✅ $url" -ForegroundColor Green
}
else {
    Write-Host "   ⏳ Pas encore disponible" -ForegroundColor Yellow
}

# Test 2: Cloud Run services
Write-Host ""
Write-Host "2️⃣  CLOUD RUN SERVICES:" -ForegroundColor Yellow
$services = gcloud run services list --project=aiprod-484120 --region=europe-west1 --format="csv[no-heading](name,status)" 2>&1
if ($services) {
    foreach ($service in $services) {
        Write-Host "   $service" -ForegroundColor White
    }
}
else {
    Write-Host "   ⏳ Pas encore créés" -ForegroundColor Yellow
}

# Test 3: Cloud SQL
Write-Host ""
Write-Host "3️⃣  CLOUD SQL INSTANCES:" -ForegroundColor Yellow
$sql = gcloud sql instances list --project=aiprod-484120 --format="csv[no-heading](name,status)" 2>&1
if ($sql) {
    foreach ($instance in $sql) {
        Write-Host "   $instance" -ForegroundColor White
    }
}
else {
    Write-Host "   ⏳ Pas encore créés" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "═════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✅ Si vous voyez des ressources: Terraform a terminé !" -ForegroundColor Green
Write-Host "⏳ Sinon, réessayez dans 5 minutes" -ForegroundColor Yellow
Write-Host "═════════════════════════════════════════════════" -ForegroundColor Cyan
