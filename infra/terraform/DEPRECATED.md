# ⚠️ DEPRECATED - Terraform IaC (Not Used Anymore)

**Status:** DEPRECATED ❌  
**Reason:** KMS infrastructure deployed via `gcloud CLI` instead  
**Date Deprecated:** February 6, 2026

---

## Why Deprecated?

Due to time constraints and 98% project completion, KMS was deployed using:

```powershell
gcloud kms keyrings create aiprod-keyring --location=global
gcloud kms keys create aiprod-key --keyring=aiprod-keyring --purpose=encryption
```

This is faster and simpler than maintaining Terraform for this single component.

---

## Alternative (Future)

If you want to manage cloud infrastructure as code in the future:

```bash
# Initialize Terraform
terraform init

# Review planned changes
terraform plan

# Apply configuration (requires Terraform binary installed)
terraform apply
```

---

## Current KMS Setup

**Status:** ✅ ACTIVE  
**Created via:** Google Cloud CLI (gcloud)  
**Date:** February 6, 2026

**Verification:**

```bash
gcloud kms keyrings list --location=global
gcloud kms keys list --keyring=aiprod-keyring --location=global
```

---

## Files in This Directory

- `main.tf` - Cloud resource definitions (not applied)
- `variables.tf` - Variable declarations (reference only)
- `outputs.tf` - Output definitions (reference only)
- `versions.tf` - Provider configuration (reference only)
- `.terraform/` - Terraform cache (can be deleted)
- `terraform.tfstate*` - State files (can be archived)

---

**Notes:** These files are kept for documentation purposes but are not actively used in the deployment pipeline.
