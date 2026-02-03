@echo off
REM Script Terraform deployment

cd /d C:\Users\averr\AIPROD_V33

REM Activer le venv
call .venv311\Scripts\activate.bat

REM Définir les variables d'env
set PATH=C:\Users\averr\AIPROD_V33\.terraform-bin;%PATH%
set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\averr\AIPROD_V33\credentials\terraform-key.json

REM Aller dans le dossier terraform
cd infra\terraform

REM Vérifier terraform
terraform --version
if errorlevel 1 (
    echo ERROR: Terraform not found!
    exit /b 1
)

echo.
echo === TERRAFORM INIT ===
terraform init

echo.
echo === TERRAFORM PLAN ===
terraform plan -out=tfplan

echo.
echo === TERRAFORM APPLY ===
terraform apply tfplan

echo.
echo === TERRAFORM OUTPUT ===
terraform output

echo.
echo ==================================
echo ✅ ÉTAPE 2 TERRAFORM COMPLÉTÉE !
echo ==================================
pause
