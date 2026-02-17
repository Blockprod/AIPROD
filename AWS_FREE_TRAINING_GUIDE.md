# 🚀 AWS Free Tier — Guide Complet pour Entraînement GRATUIT

## 💰 Offre AWS pour Nouveaux Clients

```
Crédits: $300 USD pour 12 mois
Utilisation: AWS SageMaker, EC2, etc.
Coût entraînement 17h A100: ~$70-100
Reste: $200+ inutilisé

→ TOTALEMENT GRATUIT pour vous !
```

---

## 📋 Setup Étape par Étape

### **Étape 1 : Créer compte AWS**

```
1. Aller à https://aws.amazon.com/free
2. Cliquer "Create a Free Account"
3. Email + mot de passe
4. Ajouter carte crédit (vérification, pas chargé)
5. Confirmer email
6. REÇOIT: $300 crédits automatiquement
```

**Temps : 5 min**

---

### **Étape 2 : Vérifier les Crédits**

```
1. Connecter à https://console.aws.amazon.com
2. Menu → Billing Dashboard
3. Chercher "AWS Credits"
4. Confirmer: $300 crédits actifs ✅
```

---

### **Étape 3 : Lancer EC2 avec GPU**

#### **Option A : SageMaker Notebook (Plus facile)**

```bash
1. Aller à SageMaker (search "SageMaker" dans console)
2. Notebook instances → Create notebook instance
3. Instance type: ml.p3.2xlarge (V100 32GB)
4. IAM role: Create new role
5. Create instance (~3 min startup)
6. Open JupyterLab
```

**Coût : $4.68/h (dans les crédits $300)**

---

#### **Option B : EC2 Spot GPU (Plus Économique)**

```bash
1. EC2 Console → Instances → Launch instances
2. AMI: Deep Learning AMI (Ubuntu 20.04)
3. Instance type: p3.2xlarge
4. Market options: Request Spot instance
5. Max price: $0.40/h (au lieu de $4.68/h)
6. Launch
```

**Coût : $0.40/h (économise $4/h !)**

---

### **Étape 4 : Setup AIPROD sur EC2/SageMaker**

**Une fois la machine lancée :**

```bash
# Connexion SSH (pour EC2)
ssh -i your-key.pem ubuntu@<public-ip>

# OU utilisez JupyterLab (SageMaker)
# Terminal dans JupyterLab

# Clone le repo
git clone https://github.com/Blockprod/AIPROD.git
cd AIPROD

# Install dependencies
pip install -r requirements.txt
pip install -e packages/aiprod-core
pip install -e packages/aiprod-trainer

# Vérifier GPU
nvidia-smi
# → Doit montrer V100 (32 GB) ✅
```

**Temps : 5-10 min**

---

### **Étape 5 : Télécharger les Données Requises**

```bash
# LJSpeech (2.5 GB)
cd /home/ubuntu/AIPROD
wget https://data.keithito.com/data/LJ-Speech-Dataset/LJ_Speech_Dataset.zip
unzip LJ_Speech_Dataset.zip -d data/

# Ou (depuis SageMaker terminal)
!wget https://data.keithito.com/data/LJ-Speech-Dataset/LJ_Speech_Dataset.zip
!unzip LJ_Speech_Dataset.zip -d data/
```

**Temps : 10-15 min**

---

### **Étape 6 : Lancer le Notebook Colab sur AWS**

```bash
# Copier le notebook dans Jupyter
cp notebooks/AIPROD_Sovereign_Training_Colab.ipynb ~/

# Puis dans le notebook:
# Cellule 1:
import sys
sys.path.insert(0, '/home/ubuntu/AIPROD')

# Cellule 3b:
REAL_TRAINING = True  # ← Activer mode réel

# Cellule 3d:
# Télécharger LJSpeech (ou manuellement comme ci-dessus)

# Exécuter D1a → D2 → D3 → D4 → D10
```

---

## ⏱️ Timeline d'Entraînement sur AWS

| Phase | GPU | Temps | Coût |
|-------|-----|-------|------|
| **D1a** LoRA | V100 32GB | 8h | $37.44 |
| **D2** HW-VAE | V100 32GB | 4h | $18.72 |
| **D3** Audio VAE | V100 32GB | 2h | $9.36 |
| **D4** TTS 3 phases | V100 32GB | 3h | $14.04 |
| **TOTAL** | - | **17h** | **$80 USD** |

**Crédits disponibles : $300**
**Coût réel : $80**
**Reste : $220 inutilisé** ✅

---

## 🎯 Machines AWS Recommandées

### **Pour Entraînement SHDT + VAE + TTS**

| Instance | GPU | VRAM | Coût/h | Pour |
|----------|-----|------|--------|------|
| **ml.p3.2xlarge** (SageMaker) | V100 | 32 GB | $4.68 | ✅ **Meilleur équilibre** |
| **p3.2xlarge** Spot (EC2) | V100 | 32 GB | $0.40 | 💰 **Plus économique** |
| **ml.p3.8xlarge** (SageMaker) | 4× V100 | 128 GB | $18.72 | 🚀 **Overkill mais rapide** |
| **g4dn.12xlarge** Spot (EC2) | T4 | 16 GB | $0.35 | ⚠️ T4 = Limite pour D2/D3 |

**Recommandé : p3.2xlarge Spot (~$0.40/h)**

---

## 💡 Tips pour Économiser les Crédits

### **1. Utiliser Spot Pricing (-90%)**
```bash
# Instance p3.2xlarge
# Prix normal: $4.68/h
# Spot price: $0.40/h
# Économie: $4.28/h × 17h = $73 !
```

### **2. Arrêter quand idle**
```bash
# Stop instance après entraînement
# Coût storage: ~$0.05/h (mini)
# Important: Arrêter, pas terminer (sinon perte données)
```

### **3. Utiliser S3 pour sauvegarder**
```bash
# Après entraînement:
aws s3 cp /home/ubuntu/sovereign/ s3://my-bucket/sovereign/ --recursive
# Puis terminer instance (économise stockage EBS)
```

### **4. Monitorer les coûts**
```
AWS Console → Billing → Costs & Usage
Vérifier en temps réel que sous $300
```

---

## 🔧 Commandes Utiles AWS CLI

### **Lancer instance Spot EC2 (CLI)**
```bash
aws ec2 request-spot-instances \
  --spot-price "0.40" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification '{
    "ImageId": "ami-0c55b159cbfafe1f0",
    "InstanceType": "p3.2xlarge",
    "KeyName": "your-key"
  }'
```

### **Sauvegarder vers S3**
```bash
aws s3 sync /home/ubuntu/output/ s3://my-aiprod-bucket/ --delete
```

### **Arrêter instance (pas terminer !)**
```bash
aws ec2 stop-instances --instance-ids i-xxxxxxxxx
```

---

## ✅ Checklist AWS Setup

- [ ] Créer compte AWS
- [ ] Recevoir $300 crédits ✅
- [ ] Lancer instance p3.2xlarge (SageMaker ou EC2 Spot)
- [ ] SSH/JupyterLab connexio réussie
- [ ] Clone AIPROD repo
- [ ] Installer pip packages
- [ ] Télécharger LJSpeech (~2.5 GB)
- [ ] Vérifier GPU: V100 32GB ✅
- [ ] `REAL_TRAINING = True` dans notebook
- [ ] Exécuter D1a → D2 → D3 → D4 → D10
- [ ] Télécharger `sovereign/` sur machine locale
- [ ] Arrêter instance (pas terminer!)
- [ ] Vérifier billing: ~$80 dépensé ✅

---

## 🆘 Troubleshooting

### **Problème : V100 pas disponible dans région**

**Solution :**
```
AWS Console → EC2 → Availability zones
Changer région: us-west-2 / us-east-1
(p3.2xlarge dispo dans plusieurs régions)
```

### **Problème : Spot instance interrompue**

**Solution :**
```
Utiliser on-demand price (~$4.68/h)
Ou changer région pour moins d'interruption
Coût total: ~$80 vs $0.40/h spot
```

### **Problème : Pas assez d'espace disque**

**Solution :**
```bash
# Vérifier espace
df -h

# EBS volume par défaut: 100 GB
# Suffit pour AIPROD + données + modèles

# Si besoin plus:
# AWS Console → EBS → Modify volume → Augmenter
```

### **Problème : Out of memory pendant D1a**

**Solution :**
```
V100 32GB = OK pour D1a+D2+D3+D4
Si OOM:
- Réduire batch size
- Ou upgrader vers p3.8xlarge (4× V100 = 128 GB)
```

---

## 📊 Coûts Finaux Estimés

| Scénario | Coût | Crédits |
|----------|------|---------|
| **p3.2xlarge on-demand** (17h) | $79.56 | $79.56 / $300 |
| **p3.2xlarge Spot** (17h @$0.40) | $6.80 | $6.80 / $300 |
| **Avec storage EBS** (+17h) | $0.85 | $0.85 / $300 |
| **TOTAL réaliste** | **$7-80** | **< $300** ✅ |

---

## 🎉 Résultat Final

```
✅ Entraînement COMPLET: D1a + D2 + D3 + D4
✅ Tous les modèles .safetensors générés
✅ MANIFEST.json avec SHA-256
✅ Coût: $0 (avec crédits AWS $300)
✅ Temps: ~17-20h continu (ou 2-3 jours en pause)
✅ GPU: V100 32GB (équivalent A100 40GB pour ce travail)
```

---

## 🚀 Prochaines Étapes

1. **Créer compte AWS** (~5 min)
2. **Lancer EC2/SageMaker** (~10 min)
3. **Clone + Setup** (~10 min)
4. **Entraîner** (~17-20h)
5. **Télécharger résultats** (~30 min)
6. **Arrêter instance** (economise crédits restants)

**Total billable : ~$7-80 USD** (Spot vs on-demand)
**Coûts réels pour vous : $0** (crédits AWS)

---

## 📚 Ressources

- **AWS Free Account:** https://aws.amazon.com/free/
- **AWS SageMaker Pricing:** https://aws.amazon.com/sagemaker/pricing/
- **EC2 Pricing (Spot):** https://aws.amazon.com/ec2/spot/pricing/
- **Deep Learning AMI:** https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch/
- **AWS CLI Installation:** https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

---

**✅ C'est ça la vraie solution GRATUITE !** Allez-y ! 🚀
