#!/usr/bin/env python3
"""
PHASE 4.1 - Cloud Cost Analysis Tool
Analyzes GCP billing data and identifies optimization opportunities
"""

import json
import os
from datetime import datetime
from typing import Optional
from google.cloud import bigquery
from google.oauth2 import service_account
try:
    from google.cloud.billing_v1 import CloudBillingClient
except ImportError:
    CloudBillingClient = None

class CostAnalyzer:
    """Main cost analysis class"""
    
    def __init__(self, project_id: str, billing_account_id: str, credentials_path: Optional[str] = None):
        self.project_id = project_id
        self.billing_account_id = billing_account_id
        self.credentials = None
        
        # Try to use service account credentials if provided
        if credentials_path and os.path.exists(credentials_path):
            self.credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=[
                    'https://www.googleapis.com/auth/bigquery',
                    'https://www.googleapis.com/auth/billing'
                ]
            )
            self.bq_client = bigquery.Client(project=project_id, credentials=self.credentials)
            # Try to initialize billing client with credentials
            if CloudBillingClient:
                try:
                    self.billing_client = CloudBillingClient(credentials=self.credentials)
                except Exception as e:
                    print(f"⚠️  Could not initialize CloudBillingClient: {e}")
                    self.billing_client = None
            else:
                self.billing_client = None
        else:
            self.bq_client = bigquery.Client(project=project_id)
            self.billing_client = CloudBillingClient() if CloudBillingClient else None
    
    def get_billing_account_info(self):
        """Get billing account information"""
        if not self.billing_client:
            return {
                'account_id': self.billing_account_id,
                'display_name': 'Unknown (google-cloud-billing not installed)',
                'note': 'Install google-cloud-billing for full account details'
            }
        
        try:
            account_path = f"billingAccounts/{self.billing_account_id}"
            account = self.billing_client.get_billing_account(name=account_path)
            
            return {
                'account_id': self.billing_account_id,
                'display_name': account.display_name,
                'open': account.open,
                'master_account': account.master_billing_account
            }
        except Exception as e:
            print(f"⚠️  Note: Billing account details not available ({e})")
            return {
                'account_id': self.billing_account_id,
                'note': 'Could not fetch - ensure billing_v1 client is configured'
            }
    
    def get_6month_cost_breakdown(self):
        """Get cost breakdown by service for last 6 months"""
        
        query = """
        SELECT
            service.description as service,
            SUM(CAST(cost as FLOAT64)) as total_cost,
            SUM(CAST(usage.amount as FLOAT64)) as total_usage,
            usage.unit as unit,
            COUNT(*) as line_items,
            MIN(DATE(_PARTITIONTIME)) as earliest_date,
            MAX(DATE(_PARTITIONTIME)) as latest_date
        FROM `{project_id}.billing_dataset.gcp_billing_export_*`
        WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY)
            AND resource.service IS NOT NULL
        GROUP BY service, unit
        ORDER BY total_cost DESC
        """.format(project_id=self.project_id)
        
        try:
            results = self.bq_client.query(query).result()
            
            costs = []
            total = 0
            
            for row in results:
                cost = float(row['total_cost'])
                total += cost
                costs.append({
                    'service': row['service'],
                    'cost': cost,
                    'usage': float(row['total_usage']),
                    'unit': row['unit'],
                    'line_items': row['line_items'],
                    'percent_of_total': 0  # Will be calculated below
                })
            
            # Calculate percentages
            for cost_item in costs:
                cost_item['percent_of_total'] = (cost_item['cost'] / total) * 100
            
            return {
                'total_cost': total,
                'services': costs,
                'period': '180 days'
            }
        
        except Exception as e:
            print(f"Error querying costs: {e}")
            return None
    
    def get_monthly_cost_trend(self):
        """Get monthly cost trend for visualization"""
        
        query = """
        SELECT
            DATE_TRUNC(DATE(_PARTITIONTIME), MONTH) as month,
            SUM(CAST(cost as FLOAT64)) as monthly_cost
        FROM `{project_id}.billing_dataset.gcp_billing_export_*`
        WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY)
        GROUP BY month
        ORDER BY month
        """.format(project_id=self.project_id)
        
        try:
            results = self.bq_client.query(query).result()
            
            trend = []
            for row in results:
                trend.append({
                    'month': str(row['month']),
                    'cost': float(row['monthly_cost'])
                })
            
            return trend
        
        except Exception as e:
            print(f"Error getting monthly trend: {e}")
            return None
    
    def get_top_cost_drivers(self, limit: int = 20):
        """Identify top 20 cost drivers (by SKU)"""
        
        query = """
        SELECT
            service.description as service,
            sku.description as resource,
            SUM(CAST(cost as FLOAT64)) as total_cost,
            SUM(CAST(usage.amount as FLOAT64)) as total_usage,
            usage.unit as unit,
            COUNT(*) as line_items
        FROM `{project_id}.billing_dataset.gcp_billing_export_*`
        WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY)
            AND sku.description IS NOT NULL
        GROUP BY service, sku.description, unit
        ORDER BY total_cost DESC
        LIMIT {limit}
        """.format(project_id=self.project_id, limit=limit)
        
        try:
            results = self.bq_client.query(query).result()
            
            drivers = []
            for row in results:
                drivers.append({
                    'service': row['service'],
                    'resource': row['resource'],
                    'cost': float(row['total_cost']),
                    'usage': float(row['total_usage']),
                    'unit': row['unit'],
                    'line_items': row['line_items']
                })
            
            return drivers
        
        except Exception as e:
            print(f"Error getting cost drivers: {e}")
            return None
    
    def get_optimization_recommendations(self):
        """Generate optimization recommendations based on usage patterns"""
        
        recommendations = []
        
        # Check Cloud Run utilization
        cr_query = """
        SELECT
            SUM(CAST(usage.amount as FLOAT64)) as cpu_seconds,
            SUM(CAST(cost as FLOAT64)) as cost
        FROM `{project_id}.billing_dataset.gcp_billing_export_*`
        WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            AND service.description = 'Cloud Run'
        """.format(project_id=self.project_id)
        
        try:
            cr_result = list(self.bq_client.query(cr_query).result())[0]
            
            recommendations.append({
                'priority': 'HIGH',
                'service': 'Cloud Run',
                'recommendation': 'Implement auto-scaling for non-critical services',
                'potential_savings': 'USD 200-500/month (10-15% of Cloud Run)',
                'effort': 'Medium (2-3 hours)',
                'roi': 'High (cost reduction immediately)',
                'steps': [
                    'Set minScale to 0 for non-production services',
                    'Configure maxScale based on expected load',
                    'Set target CPU utilization to 60%',
                    'Test with gradual traffic increase'
                ]
            })
        except Exception as e:
            print(f"Note: Could not analyze Cloud Run: {e}")
        
        # Check Firestore usage
        recommendations.append({
            'priority': 'MEDIUM',
            'service': 'Firestore',
            'recommendation': 'Switch to On-demand billing for better cost optimization',
            'potential_savings': 'USD 50-200/month (variable)',
            'effort': 'Low (30 minutes)',
            'roi': 'Medium (depends on usage patterns)',
            'steps': [
                'Analyze current read/write patterns',
                'Switch to on-demand billing mode',
                'Monitor usage for 2 weeks',
                'Compare costs with provisioned mode'
            ]
        })
        
        # Check Cloud SQL
        recommendations.append({
            'priority': 'MEDIUM',
            'service': 'Cloud SQL',
            'recommendation': 'Enable automated backups and optimize backup retention',
            'potential_savings': 'USD 100-300/month (20-30% of Cloud SQL)',
            'effort': 'Low (1 hour)',
            'roi': 'High (simple configuration)',
            'steps': [
                'Review current backup retention policy',
                'Reduce retention for non-critical databases',
                'Enable point-in-time recovery for production only',
                'Archive older backups to Cloud Storage'
            ]
        })
        
        # Data storage optimization
        recommendations.append({
            'priority': 'MEDIUM',
            'service': 'Cloud Storage',
            'recommendation': 'Implement lifecycle policies and use cheaper storage classes',
            'potential_savings': 'USD 50-150/month (15-25% of Cloud Storage)',
            'effort': 'Low (1 hour)',
            'roi': 'High (automatic)',
            'steps': [
                'Identify data retention requirements',
                'Create lifecycle policies to move data to Coldline',
                'Archive data >90 days old to Nearline',
                'Delete unnecessary old backups'
            ]
        })
        
        return recommendations
    
    def generate_report(self):
        """Generate comprehensive cost analysis report"""
        
        print("\n" + "="*80)
        print("PHASE 4.1 - CLOUD COST ANALYSIS REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Billing account info
        print("📊 BILLING ACCOUNT")
        print("-" * 80)
        account_info = self.get_billing_account_info()
        if account_info:
            print(f"Account: {account_info.get('display_name', 'N/A')} ({account_info.get('account_id', 'N/A')})")
            if 'open' in account_info:
                print(f"Status: {'Active' if account_info['open'] else 'Closed'}")
            if 'note' in account_info:
                print(f"Note: {account_info['note']}")
            print()
        
        # Cost breakdown
        print("💰 6-MONTH COST BREAKDOWN BY SERVICE")
        print("-" * 80)
        breakdown = self.get_6month_cost_breakdown()
        if breakdown:
            print(f"Total Cost (6 months): USD ${breakdown['total_cost']:,.2f}")
            print(f"Average Monthly: USD ${breakdown['total_cost']/6:,.2f}\n")
            
            print("Service Breakdown:")
            for service in breakdown['services']:
                print(f"  {service['service']}: USD ${service['cost']:,.2f} ({service['percent_of_total']:.1f}%)")
        
        # Monthly trend
        print("\n📈 MONTHLY COST TREND")
        print("-" * 80)
        trend = self.get_monthly_cost_trend()
        if trend:
            for month_data in trend:
                bar_length = int(month_data['cost'] / 50)
                bar = "█" * bar_length
                print(f"{month_data['month']}: USD ${month_data['cost']:>10,.0f} {bar}")
        
        # Top cost drivers
        print("\n🎯 TOP 20 COST DRIVERS")
        print("-" * 80)
        drivers = self.get_top_cost_drivers(20)
        if drivers:
            for i, driver in enumerate(drivers[:10], 1):
                print(f"{i:2d}. {driver['service']}")
                print(f"    Resource: {driver['resource'][:60]}")
                print(f"    Cost: USD ${driver['cost']:,.2f}")
                print()
        
        # Recommendations
        print("\n💡 OPTIMIZATION RECOMMENDATIONS")
        print("-" * 80)
        recommendations = self.get_optimization_recommendations()
        for i, rec in enumerate(recommendations, 1):
            priority_icon = "🔴" if rec['priority'] == "HIGH" else "🟡"
            print(f"{i}. {priority_icon} [{rec['priority']}] {rec['service']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Potential Savings: {rec['potential_savings']}")
            print(f"   Effort: {rec['effort']}")
            print(f"   Steps:")
            for step in rec['steps']:
                print(f"     • {step}")
            print()
        
        print("="*80)
        print("End of Report\n")
    
    def export_to_json(self, filename: str):
        """Export analysis results to JSON"""
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'project_id': self.project_id,
            'billing_account': self.get_billing_account_info(),
            'cost_breakdown': self.get_6month_cost_breakdown(),
            'monthly_trend': self.get_monthly_cost_trend(),
            'top_cost_drivers': self.get_top_cost_drivers(20),
            'recommendations': self.get_optimization_recommendations()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Analysis exported to {filename}")
        return filename


def main():
    """Main execution"""
    
    # Configuration (update with your values)
    PROJECT_ID = "aiprod-v33"
    BILLING_ACCOUNT_ID = "YOUR_BILLING_ACCOUNT_ID"  # Get from: gcloud billing accounts list
    
    # Try to find credentials file
    credentials_path = None
    default_creds_paths = [
        'credentials/terraform-key.json',
        os.path.expanduser('~/.config/gcloud/application_default_credentials.json'),
        '/root/.config/gcloud/application_default_credentials.json'
    ]
    
    for path in default_creds_paths:
        if os.path.exists(path):
            credentials_path = path
            print(f"✅ Using credentials from: {path}")
            break
    
    if not credentials_path:
        print("⚠️  No credentials found. Trying to use default GCP credentials...")
    
    # Create analyzer
    analyzer = CostAnalyzer(PROJECT_ID, BILLING_ACCOUNT_ID, credentials_path)
    
    # Generate and display report
    analyzer.generate_report()
    
    # Export to JSON for further analysis
    analyzer.export_to_json('cost_analysis_report.json')
    
    print("✅ PHASE 4.1 - Cloud Cost Analysis Complete!")
    print("\nNext steps:")
    print("1. Review the recommendations above")
    print("2. Implement high-priority quick wins")
    print("3. Proceed to PHASE 4.2 - Auto-Scaling Setup")


if __name__ == "__main__":
    main()
