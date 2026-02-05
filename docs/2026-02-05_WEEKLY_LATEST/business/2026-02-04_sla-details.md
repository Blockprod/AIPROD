# Service Level Agreement (SLA) — AIPROD

**Effective Date**: February 4, 2026  
**Last Updated**: February 4, 2026  
**Version**: 1.0  
**Status**: 🟢 Active

---

## Executive Summary

AIPROD commits to industry-leading service levels across all customer tiers. This document defines uptime guarantees, response time targets, support levels, and credit policies.

**Key Commitments**:

- 🟢 Free Tier: 99% uptime (best effort)
- 🟡 Pro Tier: 99.5% uptime (guaranteed)
- 🔴 Enterprise Tier: 99.95% uptime (guaranteed)

---

## 1. Uptime SLA

### Definitions

**Uptime**: Percentage of time when AIPROD API is accessible and responsive.

**Downtime**: When AIPROD API returns 5xx errors or is completely unavailable for users with valid credentials.

**Downtime Window**: Measured in 5-minute intervals. Service must be down for entire window to count as downtime.

**Calculation Formula**:

```
Uptime % = (Total Minutes - Downtime Minutes) / Total Minutes × 100
```

### Uptime Targets

| Tier       | Uptime Target | Monthly Downtime Allowed | SLA Credit |
| ---------- | ------------- | ------------------------ | ---------- |
| Free       | 99%           | 7.2 hours                | None       |
| Pro        | 99.5%         | 3.6 hours                | 10%        |
| Enterprise | 99.95%        | 21.6 minutes             | 20%        |

### Uptime Calculation Details

**Per Month (30 days = 43,200 minutes)**:

- **99% SLA** = 428 minutes downtime allowed
- **99.5% SLA** = 216 minutes downtime allowed
- **99.95% SLA** = 21.6 minutes downtime allowed

**Per Year (365 days)**:

- **99% SLA** = 3.65 days downtime allowed
- **99.5% SLA** = 1.83 days downtime allowed
- **99.95% SLA** = 43.2 minutes downtime allowed

### Uptime Exclusions

The following do NOT count toward SLA violations:

1. **Scheduled Maintenance**
   - Announced 7+ days in advance
   - Limited to 4 hours per calendar month
   - Typically scheduled: Saturday 2-4 AM UTC

2. **Customer-Caused Outages**
   - Invalid API credentials or quota exceeded
   - Misconfigurations in customer code
   - Abuse or policy violations

3. **Third-Party Service Failures**
   - Video rendering backends (Runway, Veo-3) unavailable
   - External API dependencies (Google, cloud services)
   - DNS or CDN failures outside our control

4. **Force Majeure**
   - Natural disasters, wars, terrorism
   - Government actions or regulations
   - Pandemics or public health emergencies

5. **Known Limitations** (Free Tier only)
   - Free tier has NO uptime guarantee
   - Best effort only, no credits

---

## 2. Response Time SLA

### Definition

**Response Time**: Time from when AIPROD receives request until we send response headers.

**P95**: 95th percentile (95% of requests faster than this)  
**P99**: 99th percentile (99% of requests faster than this)

### Response Time Targets

| Endpoint Type     | Free   | Pro    | Enterprise |
| ----------------- | ------ | ------ | ---------- |
| Health checks     | <100ms | <50ms  | <20ms      |
| Authentication    | <200ms | <100ms | <50ms      |
| Job submission    | <500ms | <300ms | <200ms     |
| Job status query  | <300ms | <200ms | <100ms     |
| Results retrieval | <1s    | <500ms | <300ms     |
| Batch operations  | <3s    | <2s    | <1s        |

### Response Time Measurement

- Measured from our API gateway perspective
- P95 latency is primary SLA metric
- Excludes network latency (customer's ISP)
- Excludes time spent in customer-provided callbacks

### Response Time Credits

| Metric      | Threshold | Free | Pro         | Enterprise  |
| ----------- | --------- | ---- | ----------- | ----------- |
| P95 Latency | >2s       | N/A  | Monitor     | Monitor     |
| P95 Latency | >5s       | N/A  | Alert       | Alert       |
| P99 Latency | >10s      | N/A  | Investigate | Investigate |

---

## 3. Job Processing Time SLA

### Definition

**Processing Time**: Time from job submission until results available (excluding rendering backend time).

### Processing Time Targets

| Scenario                     | Free       | Pro       | Enterprise |
| ---------------------------- | ---------- | --------- | ---------- |
| Simple processing (no video) | 1-2 min    | 1-2 min   | 30-60 sec  |
| Standard video (<5 min)      | 15-30 min  | 10-20 min | 5-15 min   |
| Long video (5-30 min)        | 30-120 min | 20-60 min | 15-45 min  |
| With complex effects         | +50%       | +30%      | +20%       |

### Job Processing Exclusions

The following do NOT impact processing time SLA:

1. **Rendering Backend Delays**: Time spent in external video rendering services
2. **User-Initiated Delays**: Custom preprocessing or callbacks
3. **Queue Wait Time**: Wait to start processing (depends on load)

---

## 4. Support SLA

### Support Channels by Tier

| Tier       | Email | Chat | Phone | Slack |
| ---------- | ----- | ---- | ----- | ----- |
| Free       | Yes   | No   | No    | No    |
| Pro        | Yes   | Yes  | No    | Yes   |
| Enterprise | Yes   | Yes  | Yes   | Yes   |

### Support Response Times

| Tier       | Severity | Target Response | Target Resolution |
| ---------- | -------- | --------------- | ----------------- |
| Free       | All      | 48 hours        | N/A               |
| Pro        | Critical | 1 hour          | 8 hours           |
| Pro        | High     | 4 hours         | 24 hours          |
| Pro        | Normal   | 24 hours        | 3 days            |
| Enterprise | Critical | 15 minutes      | 1 hour            |
| Enterprise | High     | 1 hour          | 4 hours           |
| Enterprise | Normal   | 4 hours         | 24 hours          |

### Support Severity Levels

**Critical (P1)**

- Entire service unavailable or unusable
- Recurring errors affecting all jobs
- Complete data loss or corruption
- Security vulnerability

**High (P2)**

- Major feature broken (can't submit jobs)
- Significant performance degradation
- Intermittent errors
- Limited workaround available

**Normal (P3)**

- Feature partially working
- Single feature broken with workaround
- Documentation request
- Feature inquiry

---

## 5. Monthly Downtime & Service Credits

### Service Credit Schedule

When we fail to meet SLA, you receive automatic credits:

| Tier       | <0.1% Downtime | 0.1-0.5% | 0.5-1% | 1-5% | >5%  |
| ---------- | -------------- | -------- | ------ | ---- | ---- |
| Free       | N/A            | N/A      | N/A    | N/A  | N/A  |
| Pro        | No credit      | 10%      | 25%    | 50%  | 100% |
| Enterprise | No credit      | 10%      | 25%    | 50%  | 100% |

### Credit Examples

**Example 1: Pro Tier, 0.3% Downtime**

- Monthly bill: $299
- Credit: 10% = $29.90
- Next invoice: $269.10

**Example 2: Enterprise Tier, 2% Downtime**

- Monthly bill: $999
- Credit: 50% = $499.50
- Next invoice: $499.50

**Example 3: Free Tier (No SLA)**

- No credits offered
- Service is best effort only

### How Credits Are Applied

1. **Automatic**: Credits calculated and applied automatically
2. **Timing**: Applied to invoice in month following incident
3. **Cap**: Total credits per month cannot exceed 100% of bill
4. **Non-Refundable**: Credits do not result in cash refunds
5. **Non-Transferable**: Credits cannot be transferred to other accounts

### Manual Credit Requests

If automatic credits not applied:

1. File request within 30 days of incident
2. Provide:
   - Account name and ID
   - Incident date/time
   - Impact description (jobs affected, revenue loss)
3. Expected response: 5 business days

---

## 6. Tier Comparison

### Free Tier

| Feature                | Free                          |
| ---------------------- | ----------------------------- |
| **Uptime SLA**         | 99% (best effort, no credits) |
| **Response Time SLA**  | No guarantee                  |
| **Support SLA**        | No response time guarantee    |
| **Support Channel**    | Email only                    |
| **Rate Limit**         | 10 requests/minute            |
| **Concurrent Jobs**    | 2                             |
| **Storage**            | 10 GB                         |
| **Data Retention**     | 7 days                        |
| **Status Page Access** | Public only                   |

### Pro Tier

| Feature                | Pro                     |
| ---------------------- | ----------------------- |
| **Uptime SLA**         | 99.5% guaranteed        |
| **Response Time SLA**  | <300ms P95              |
| **Support SLA**        | 4-hour response         |
| **Support Channels**   | Email, Chat, Slack      |
| **Rate Limit**         | 100 requests/minute     |
| **Concurrent Jobs**    | 10                      |
| **Storage**            | 500 GB                  |
| **Data Retention**     | 90 days                 |
| **Status Page Access** | Detailed + email alerts |
| **Webhooks**           | ✅ Supported            |
| **Custom Presets**     | Up to 3                 |
| **Priority Support**   | ✅ Yes                  |

### Enterprise Tier

| Feature                | Enterprise                      |
| ---------------------- | ------------------------------- |
| **Uptime SLA**         | 99.95% guaranteed               |
| **Response Time SLA**  | <200ms P95                      |
| **Support SLA**        | 15-minute response              |
| **Support Channels**   | Email, Chat, Slack, Phone       |
| **Dedicated Contact**  | TAM (Technical Account Manager) |
| **Rate Limit**         | Unlimited                       |
| **Concurrent Jobs**    | 50                              |
| **Storage**            | 2 TB                            |
| **Data Retention**     | 1 year                          |
| **Status Page Access** | Premium + custom dashboard      |
| **Webhooks**           | ✅ Supported with retry         |
| **Custom Presets**     | Unlimited                       |
| **SLA Review**         | Quarterly                       |

---

## 7. Monitoring & Transparency

### Status Page

AIPROD provides real-time status at: https://status.aiprod.ai

**Displays**:

- Current system status
- Incident history
- Scheduled maintenance
- Component status (API, Database, Storage)
- Uptime statistics (30/90 days)

### Monitoring Tools

- **Real-time Dashboard**: Available to Pro+ customers
- **Weekly Reports**: Email summary of metrics
- **Custom Alerts**: Enterprise can set up custom alert thresholds

### Performance Metrics

We publicly report:

1. **API Availability**: Percentage of successful requests
2. **Response Time**: P50, P95, P99 latencies
3. **Error Rate**: Percentage of 5xx errors
4. **Job Success Rate**: Percentage of completed jobs
5. **Database Uptime**: Cloud SQL instance availability

**Reported Monthly**: Metrics posted to status page by 5th of month

---

## 8. Maintenance Windows

### Scheduled Maintenance

**Policy**:

- Maximum 4 hours per calendar month
- Announced 7+ days in advance
- Typical window: Saturday 2-4 AM UTC
- No SLA credit for announced maintenance

### Emergency Maintenance

**Policy**:

- May be needed with <24 hour notice
- Counts against SLA if unplanned
- Will trigger credits if SLA breached

### Maintenance Notifications

Customers notified via:

1. Status page banner (7 days before)
2. Email alert (if subscribed)
3. In-app notification (if logged in)
4. Twitter/status updates

---

## 9. Scope & Exclusions

### What's Covered by SLA

✅ AIPROD API availability and response time  
✅ Cloud SQL database uptime  
✅ Cloud Run service availability  
✅ Cloud Storage service availability

### What's NOT Covered by SLA

❌ Third-party services (Runway, Veo-3, Google Vertex AI)  
❌ Customer network/internet connectivity  
❌ Client-side errors in customer code  
❌ Custom integrations or webhooks  
❌ Free tier service (best effort only)  
❌ Beta/preview features  
❌ Abuse or policy violations

---

## 10. Dispute Resolution

### SLA Violation Dispute Process

1. **Report**: Contact support@aiprod.ai with:
   - Account ID and tier
   - Dates/times of incident
   - Impact evidence (logs, screenshots)

2. **Investigation**: Our team investigates within 5 business days

3. **Response**: We provide:
   - Root cause analysis
   - SLA impact determination
   - Credit amount (if applicable)

4. **Resolution**: Credits applied or dispute explained

### Escalation

If unsatisfied with resolution:

- **Pro Tier**: Escalate to service@aiprod.ai
- **Enterprise Tier**: Escalate to TAM and legal@aiprod.ai
- **All Tiers**: May pursue arbitration per Terms of Service

---

## 11. Changes to SLA

### Modification Policy

- Minimum 30 days notice for changes
- Changes effective on next billing cycle
- Existing commitments honored for current term
- Improvement changes effective immediately

### Version History

| Version | Date        | Changes              |
| ------- | ----------- | -------------------- |
| 1.0     | Feb 4, 2026 | Initial SLA document |

---

## 12. Contact & Support

### Support Portal

- **URL**: https://support.aiprod.ai
- **Email**: support@aiprod.ai
- **Chat** (Pro+): https://dashboard.aiprod.ai/chat
- **Phone** (Enterprise): +1-XXX-XXX-XXXX

### For SLA Questions

- **Email**: sla@aiprod.ai
- **Slack Channel** (Enterprise): #your-account-sla
- **dedicated TAM** (Enterprise): [Your contact]

### Emergency Contacts

- **Incident Management**: incidents@aiprod.ai
- **Security Issues**: security@aiprod.ai
- **Billing/Account**: billing@aiprod.ai

---

## Appendix A: Calculating Credits

### Step 1: Measure Downtime

```bash
# Check our logs for the month
downtime_minutes = total_error_minutes_from_our_systems

# Example: 108 minutes downtime in 43,200 minute month
# = (108 / 43,200) × 100 = 0.25% downtime
```

### Step 2: Determine Credit Tier

```
0.25% downtime falls in 0.1-0.5% bracket
Pro tier credit: 10% of monthly bill
```

### Step 3: Calculate Credit Amount

```
Monthly bill: $299
Credit: 10% × $299 = $29.90
```

### Step 4: Apply to Invoice

```
Next month's invoice: $299 - $29.90 = $269.10
```

---

## Appendix B: Monitoring Dashboard Access

### For Pro Tier

- Login to https://dashboard.aiprod.ai
- Go to Settings → Monitoring
- View: Uptime, Response Time, Error Rate (last 30 days)

### For Enterprise Tier

- Custom dashboard with:
  - Real-time metrics
  - Custom time ranges
  - Comparative analysis
  - Alert configuration
  - SLA tracking
  - Trend analysis

---

## Appendix C: Terms Definitions

| Term              | Definition                                            |
| ----------------- | ----------------------------------------------------- |
| **Uptime**        | Percentage of time API responds without 5xx errors    |
| **Response Time** | Latency from request receipt to response sent         |
| **Availability**  | Service is reachable and processing requests          |
| **Incident**      | Any event causing service degradation                 |
| **Downtime**      | Period when service fails to meet availability target |
| **SLA**           | Service Level Agreement; binding commitment           |
| **Credit**        | Account credit applied as discount on next invoice    |
| **TAM**           | Technical Account Manager (Enterprise only)           |

---

## Appendix D: FAQ

**Q: What if I'm on the wrong tier?**
A: You can upgrade anytime via dashboard. Downgrade takes effect at end of billing period.

**Q: Can I get a refund instead of credit?**
A: No, credits are only form of compensation. Non-refundable per Terms of Service.

**Q: Does data transfer count toward SLA?**
A: No, data transfer is not part of availability or response time SLAs.

**Q: What about performance during spikes?**
A: SLA applies during all times, including traffic spikes. We auto-scale to maintain SLA.

**Q: Are there any guaranteed response times for batch jobs?**
A: No, batch processing is best-effort. Individual API calls still have response time SLAs.

**Q: Can I get a custom SLA?**
A: Enterprise customers can discuss custom SLAs with their TAM.

---

**Document Status**: 🟢 Active  
**Last Updated**: February 4, 2026  
**Next Review**: May 4, 2026  
**Owner**: Legal + Engineering  
**Version**: 1.0
