# 🔐 AIPROD Environment Validation Script (PowerShell)
# Validates environment variables and security

param(
    [switch]$Strict = $false,
    [switch]$Fix = $false
)

Write-Host "🔐 AIPROD Security & Environment Validator" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$PASSED = 0
$FAILED = 0
$WARNINGS = 0

function Check-Variable {
    param(
        [string]$VarName,
        [bool]$IsCritical = $false
    )
    
    $varValue = [Environment]::GetEnvironmentVariable($VarName)
    
    if ([string]::IsNullOrEmpty($varValue)) {
        if ($IsCritical) {
            Write-Host "❌ $VarName is not set (CRITICAL)" -ForegroundColor Red
            $script:FAILED++
        }
        else {
            Write-Host "⚠️  $VarName not configured (optional)" -ForegroundColor Yellow
            $script:WARNINGS++
        }
    }
    elseif ($varValue -match "^your-|^test-|^path/to/") {
        Write-Host "❌ $VarName has placeholder value" -ForegroundColor Red
        $script:FAILED++
    }
    else {
        $masked = $varValue.Substring(0, [Math]::Min(4, $varValue.Length)) + "****..."
        Write-Host "✅ $VarName configured" -ForegroundColor Green
        $script:PASSED++
    }
}

function Scan-Hardcoded-Secrets {
    Write-Host ""
    Write-Host "Scanning for hardcoded secrets..." -ForegroundColor Yellow
    
    $patterns = @{
        "GEMINI_API_KEY"    = "AIzaSy[A-Za-z0-9_-]{40,}"
        "RUNWAY_API_KEY"    = "key_[a-f0-9]{60,}"
        "Database_Password" = "password\s*[=:]\s*(['\"]?)([^'\";\s]+)\1"
    }
    
    $codeFiles = Get-ChildItem -Path "src", "scripts" -Filter "*.py" -Recurse -ErrorAction SilentlyContinue
    $secretsFound = $false
    
    foreach ($file in $codeFiles) {
        $content = Get-Content $file.FullName -ErrorAction SilentlyContinue
        
        foreach ($patternName in $patterns.Keys) {
            if ($content -match $patterns[$patternName]) {
                Write-Host "⚠️  Potential secret found in $($file.Name): $patternName" -ForegroundColor Red
                $secretsFound = $true
                $script:FAILED++
            }
        }
    }
    
    if (-not $secretsFound) {
        Write-Host "✅ No hardcoded secrets detected" -ForegroundColor Green
    }
}

Write-Host "Critical Variables:" -ForegroundColor Cyan
Write-Host "------------------"
Check-Variable "GOOGLE_CLOUD_PROJECT" $true
Check-Variable "DATABASE_URL" $true
Check-Variable "GEMINI_API_KEY" $true
Check-Variable "RUNWAY_API_KEY" $true

Write-Host ""
Write-Host "Optional Variables:" -ForegroundColor Cyan
Write-Host "------------------"
Check-Variable "REPLICATE_API_KEY" $false
Check-Variable "ELEVENLABS_API_KEY" $false
Check-Variable "FIREBASE_CREDENTIALS" $false

Write-Host ""
Write-Host "Environment Variables Check:" -ForegroundColor Cyan
Write-Host "----------------------------"

# Check for consistent naming
$hasOldNames = $false
if ([Environment]::GetEnvironmentVariable("GCP_PROJECT_ID")) {
    Write-Host "⚠️  Found deprecated GCP_PROJECT_ID - use GOOGLE_CLOUD_PROJECT instead" -ForegroundColor Yellow
    $hasOldNames = $true
    $script:WARNINGS++
}

if ([Environment]::GetEnvironmentVariable("REPLICATE_API_TOKEN")) {
    Write-Host "⚠️  Found deprecated REPLICATE_API_TOKEN - use REPLICATE_API_KEY instead" -ForegroundColor Yellow
    $hasOldNames = $true
    $script:WARNINGS++
}

if ([Environment]::GetEnvironmentVariable("RUNWAYML_API_SECRET")) {
    Write-Host "⚠️  Found deprecated RUNWAYML_API_SECRET - use RUNWAY_API_KEY instead" -ForegroundColor Yellow
    $hasOldNames = $true
    $script:WARNINGS++
}

if (-not $hasOldNames) {
    Write-Host "✅ Using standardized environment variable names" -ForegroundColor Green
}

# Scan for hardcoded secrets if requested
if ($Strict) {
    Scan-Hardcoded-Secrets
}

# Print summary
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  ✅ Passed: $PASSED" -ForegroundColor Green
Write-Host "  ❌ Failed: $FAILED" -ForegroundColor Red
if ($WARNINGS -gt 0) {
    Write-Host "  ⚠️  Warnings: $WARNINGS" -ForegroundColor Yellow
}
Write-Host "==========================================" -ForegroundColor Cyan

if ($FAILED -gt 0) {
    Write-Host ""
    Write-Host "❌ Environment validation FAILED - Please fix the issues above" -ForegroundColor Red
    exit 1
} elseif ($WARNINGS -gt 0 -and $Strict) {
    Write-Host ""
    Write-Host "⚠️  Environment validation PASSED with warnings" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host ""
    Write-Host "✅ Environment validation PASSED" -ForegroundColor Green
    exit 0
}
