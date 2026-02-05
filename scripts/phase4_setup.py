#!/usr/bin/env python3
"""
PHASE 4 Setup Script
Prepares environment for PHASE 4 tasks
"""

import os
import sys
import subprocess
from pathlib import Path

class Phase4Setup:
    """Setup helper for PHASE 4"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docs_path = self.project_root / "docs" / "plans"
        self.scripts_path = self.project_root / "scripts"
    
    def check_gcloud_setup(self):
        """Verify gcloud CLI is installed and configured"""
        print("\n🔍 Checking GCP Setup...")
        
        try:
            result = subprocess.run(
                ["gcloud", "config", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("✅ gcloud CLI is installed and configured")
                return True
            else:
                print("❌ gcloud CLI not properly configured")
                print("   Run: gcloud auth login")
                return False
        except FileNotFoundError:
            print("❌ gcloud CLI not found")
            print("   Install: https://cloud.google.com/sdk/docs/install")
            return False
    
    def check_bigquery_setup(self):
        """Verify BigQuery dataset exists for billing data"""
        print("\n🔍 Checking BigQuery Setup...")
        
        try:
            result = subprocess.run(
                ["bq", "ls", "-d", "--project_id=aiprod-v33"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if "billing_dataset" in result.stdout:
                print("✅ BigQuery billing_dataset exists")
                return True
            else:
                print("⚠️  BigQuery billing_dataset not found")
                print("\n   To create it:")
                print("   bq mk --dataset --location=US aiprod-v33:billing_dataset")
                print("\n   To export GCP billing data:")
                print("   1. Go to GCP Console > Billing")
                print("   2. Click 'Billing export to BigQuery'")
                print("   3. Select billing_dataset as destination")
                return False
        except FileNotFoundError:
            print("⚠️  bq CLI not found (install google-cloud-bigquery)")
            return False
    
    def check_python_dependencies(self):
        """Verify required Python packages"""
        print("\n🔍 Checking Python Dependencies...")
        
        required_packages = {
            'google.cloud': 'google-cloud-bigquery',
            'google.cloud.billing_v1': 'google-cloud-billing',
            'pandas': 'pandas'
        }
        
        missing_packages = []
        
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"✅ {package_name}")
            except ImportError:
                print(f"❌ {package_name} not installed")
                missing_packages.append(package_name)
        
        if missing_packages:
            print("\n   To install missing packages:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
        
        return True
    
    def verify_documentation(self):
        """Verify PHASE 4 documentation files exist"""
        print("\n🔍 Checking Documentation...")
        
        required_files = [
            "2026-02-04_PHASE4_ADVANCED_OPTIMIZATION.md",
            "2026-02-04_PHASE4_README.md"
        ]
        
        all_exist = True
        for filename in required_files:
            filepath = self.docs_path / filename
            if filepath.exists():
                print(f"✅ {filename}")
            else:
                print(f"❌ {filename} not found at {filepath}")
                all_exist = False
        
        return all_exist
    
    def verify_scripts(self):
        """Verify PHASE 4 scripts exist"""
        print("\n🔍 Checking Scripts...")
        
        required_scripts = [
            "phase4_cost_analyzer.py"
        ]
        
        all_exist = True
        for script_name in required_scripts:
            script_path = self.scripts_path / script_name
            if script_path.exists():
                print(f"✅ {script_name}")
            else:
                print(f"❌ {script_name} not found at {script_path}")
                all_exist = False
        
        return all_exist
    
    def print_summary(self, gcloud_ok, bq_ok, python_ok, docs_ok, scripts_ok):
        """Print setup summary"""
        print("\n" + "="*80)
        print("PHASE 4 SETUP SUMMARY")
        print("="*80)
        
        status = {
            'GCP Setup': '✅' if gcloud_ok else '❌',
            'BigQuery': '✅' if bq_ok else '⚠️',
            'Python Dependencies': '✅' if python_ok else '❌',
            'Documentation': '✅' if docs_ok else '❌',
            'Scripts': '✅' if scripts_ok else '❌'
        }
        
        for component, icon in status.items():
            print(f"{icon} {component}")
        
        all_ok = all([gcloud_ok, python_ok, docs_ok, scripts_ok])
        
        print("\n" + "="*80)
        if all_ok:
            print("✅ PHASE 4 SETUP COMPLETE!")
            print("\nYou can now run:")
            print("  python3 scripts/phase4_cost_analyzer.py")
            print("\nRefer to documentation:")
            print("  docs/plans/2026-02-04_PHASE4_README.md")
        else:
            print("⚠️  PHASE 4 SETUP INCOMPLETE")
            print("\nPlease fix the issues above before proceeding.")
        
        print("="*80 + "\n")
    
    def run(self):
        """Run all checks"""
        print("\n" + "="*80)
        print("PHASE 4 SETUP VERIFICATION")
        print("="*80)
        
        gcloud_ok = self.check_gcloud_setup()
        bq_ok = self.check_bigquery_setup()
        python_ok = self.check_python_dependencies()
        docs_ok = self.verify_documentation()
        scripts_ok = self.verify_scripts()
        
        self.print_summary(gcloud_ok, bq_ok, python_ok, docs_ok, scripts_ok)
        
        return gcloud_ok and python_ok and docs_ok and scripts_ok


def main():
    """Main entry point"""
    setup = Phase4Setup()
    success = setup.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
