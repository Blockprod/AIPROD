# AIPROD V33 - Comprehensive Troubleshooting Guide

**Last Updated**: February 4, 2026  
**Version**: 2.0  
**Audience**: Developers, System Administrators, Support Teams

---

## Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Authentication Issues](#authentication-issues)
3. [Request/Response Problems](#requestresponse-problems)
4. [Webhook Issues](#webhook-issues)
5. [Rate Limiting & Quotas](#rate-limiting--quotas)
6. [Data Processing Issues](#data-processing-issues)
7. [Database Problems](#database-problems)
8. [Cloud Infrastructure Issues](#cloud-infrastructure-issues)
9. [Performance Issues](#performance-issues)
10. [Security & Access Issues](#security--access-issues)
11. [Integration Troubleshooting](#integration-troubleshooting)
12. [FAQ & Advanced Diagnostics](#faq--advanced-diagnostics)

---

## Quick Diagnosis

### Symptom Flowchart

```
Is the API responding?
├─ NO → [API Unresponsive](#api-unresponsive)
└─ YES
   ├─ Status 4xx → [Client Errors](#client-errors)
   ├─ Status 5xx → [Server Errors](#server-errors)
   └─ Slow Response → [Performance Issues](#performance-issues)
```

### Diagnostic Commands

**Check API Health**:

```bash
# Quick health check
curl -s https://api.aiprod.com/health | jq .
# Expected: {"status": "healthy", "version": "33.x.x", "timestamp": "..."}

# Detailed diagnostics
curl -s -H "X-API-Key: YOUR_KEY" https://api.aiprod.com/diagnostics | jq .
# Returns: uptime, version, database status, external service status
```

**Log Collection for Support**:

```bash
# Get last 100 API errors from your account
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/logs?filter=error&limit=100&format=json" | jq > logs.json

# Include in support tickets:
# 1. Request ID (X-Request-ID header from response)
# 2. Timestamp (ISO 8601)
# 3. Operation performed
# 4. Full error response
```

---

## Authentication Issues

### Issue: 401 Unauthorized

**Symptoms**:

- All requests return `401 Unauthorized`
- Error: `"Invalid or expired credentials"`
- Headers show `WWW-Authenticate: Bearer realm="AIPROD API"`

**Root Causes & Solutions**:

| Cause                   | Detection                                | Fix                                                       |
| ----------------------- | ---------------------------------------- | --------------------------------------------------------- |
| **Missing API Key**     | No `Authorization` or `X-API-Key` header | Add header: `curl -H "X-API-Key: sk_live_xxx" ...`        |
| **Expired API Key**     | Key created >2 years ago                 | Generate new key: Account → API Keys → New Key            |
| **Revoked Token**       | Key used after revocation                | Check key status at Account → API Keys                    |
| **Wrong Environment**   | Using sandbox key in production          | Use live key: `sk_live_*` (not `sk_test_*`)               |
| **Malformed Header**    | Header syntax incorrect                  | Use: `Authorization: Bearer token` or `X-API-Key: key`    |
| **Trailing Whitespace** | Extra spaces in key                      | Trim whitespace: `key.strip()` (Python) or `.trim()` (JS) |

**Step-by-Step Fix**:

```python
# ❌ WRONG
import requests
headers = {"X-API-Key": "sk_live_abc123 "}  # Trailing space!
response = requests.get("https://api.aiprod.com/jobs", headers=headers)
# → 401 Unauthorized

# ✅ CORRECT
import requests
api_key = "sk_live_abc123".strip()  # Remove whitespace
headers = {"X-API-Key": api_key}
response = requests.get("https://api.aiprod.com/jobs", headers=headers)
# → 200 OK
```

**Prevention**:

- Store API keys in environment variables, never in code
- Use credential managers (AWS Secrets, GCP Secret Manager)
- Rotate keys annually
- Monitor key usage for suspicious activity

---

### Issue: 403 Forbidden

**Symptoms**:

- Request is authenticated but returns `403 Forbidden`
- Error: `"Insufficient permissions for this operation"`
- Different from 401 — key is valid but lacks scope

**Root Causes & Solutions**:

| Cause                   | Detection                                           | Fix                                |
| ----------------------- | --------------------------------------------------- | ---------------------------------- |
| **Insufficient Scope**  | Key lacks required permission                       | Regenerate key with broader scopes |
| **Tier Limitation**     | Feature requires Pro/Enterprise                     | Upgrade account tier               |
| **Rate Limit Exceeded** | See [Rate Limiting](#rate-limiting--quotas) section | Wait or upgrade tier               |
| **IP Whitelist**        | IP restriction enabled                              | Add your IP to whitelist           |
| **Account Suspended**   | Account inactive or in violation                    | Contact support@aiprod.com         |

**Check Your Key Permissions**:

```bash
# Get current key scopes
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/api-keys/current" | jq .scopes
# Expected output:
# {
#   "scopes": ["jobs.read", "jobs.create", "jobs.delete", "webhooks.manage"],
#   "tier": "pro",
#   "rate_limit_requests": 10000,
#   "rate_limit_period": "month"
# }
```

---

## Request/Response Problems

### Issue: 400 Bad Request

**Symptoms**:

- Request rejected immediately
- Error details in response body: `"field_name": ["error message"]`
- Example: `{"name": ["Field is required"]}`

**Common Validation Errors**:

```json
// ❌ Missing required field
{
  "error": "Validation failed",
  "details": {
    "name": ["This field is required"]
  }
}
// ✅ Fix: Include name field
POST /jobs {
  "name": "My Job",
  "input_url": "https://..."
}

// ❌ Invalid field type
{
  "error": "Validation failed",
  "details": {
    "timeout": ["Expected integer, got string"]
  }
}
// ✅ Fix: Use correct type
POST /jobs {
  "name": "My Job",
  "timeout": 3600  // integer, not "3600" string
}

// ❌ Invalid enum value
{
  "error": "Validation failed",
  "details": {
    "status": ["Must be one of: pending, running, completed, failed"]
  }
}
// ✅ Fix: Use valid value
PATCH /jobs/123 {
  "status": "completed"  // valid status
}
```

**Field Validation Reference**:

```
name (string, required)
  - Min 1 character, max 255
  - Example: "Extract podcast transcript"

timeout (integer, optional)
  - Min 60, max 7200 (seconds)
  - Default: 3600
  - Example: 1800

webhook_url (string, optional)
  - Must be valid HTTPS URL
  - Must include protocol: https://
  - Example: "https://your-domain.com/webhook"

metadata (object, optional)
  - Max 100 keys
  - Values must be strings, max 1000 chars
  - Example: {"source": "website", "user_id": "123"}
```

**Debugging 400 Errors**:

```python
import requests
import json

response = requests.post(
    "https://api.aiprod.com/jobs",
    headers={"X-API-Key": "YOUR_KEY"},
    json={
        "name": "Test",
        # Missing other required fields
    }
)

if response.status_code == 400:
    errors = response.json()
    print("Validation Errors:")
    for field, error_list in errors.get("details", {}).items():
        print(f"  {field}: {', '.join(error_list)}")
    # Output:
    # Validation Errors:
    #   input_url: This field is required
    #   type: Unknown choice
```

---

### Issue: 422 Unprocessable Entity

**Symptoms**:

- Request format is valid but business logic rejects it
- Error: `"Cannot create job: invalid input URL"` or similar
- Different from 400 — syntax is OK, content is problematic

**Common 422 Scenarios**:

```json
// ❌ URL unreachable
{
  "error": "Unprocessable entity",
  "code": "INVALID_URL",
  "message": "Input URL is unreachable: connection timeout after 10s"
}
// ✅ Fix: Ensure URL is:
//   - Publicly accessible (no private IPs)
//   - Returns content quickly (<30s)
//   - HTTPS (not HTTP)
//   - Returns 2xx status code

// ❌ Insufficient quota
{
  "error": "Unprocessable entity",
  "code": "QUOTA_EXCEEDED",
  "message": "Monthly job limit (100) exceeded. Upgrade to Pro for 10,000/month."
}
// ✅ Fix: Upgrade plan or wait for quota reset (monthly)

// ❌ Duplicate job
{
  "error": "Unprocessable entity",
  "code": "DUPLICATE_JOB",
  "message": "Job with this name already exists in this month"
}
// ✅ Fix: Use unique job name or delete old job
```

---

## Webhook Issues

### Issue: Webhooks Not Firing

**Symptoms**:

- Job completes but webhook never called
- No error in job status
- Webhook URL was verified

**Diagnosis Steps**:

**Step 1: Verify Webhook Endpoint**:

```bash
# Test your webhook manually
curl -X POST https://your-domain.com/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "event": "job.completed",
    "data": {"id": "job_123", "status": "completed"}
  }' \
  -v

# Check response:
# - Status 200/201/202 = SUCCESS
# - Status 3xx = REDIRECT (may fail)
# - Status 4xx/5xx = ERROR
# - Timeout > 10s = FAILURE

# If returning error, fix endpoint first before retesting webhooks
```

**Step 2: Check Webhook Logs**:

```bash
# Get webhook delivery history
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/webhooks?include_logs=true" | jq '.[].logs[-5:]'

# Output:
# [
#   {
#     "timestamp": "2026-02-04T10:30:45Z",
#     "event": "job.completed",
#     "status": "delivered",
#     "response_code": 200
#   },
#   {
#     "timestamp": "2026-02-04T10:20:15Z",
#     "event": "job.failed",
#     "status": "failed",
#     "response_code": 500,
#     "error": "Internal Server Error"
#   }
# ]
```

**Step 3: Check Registration**:

```bash
# List registered webhooks
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/webhooks" | jq .
# Expected:
# [
#   {
#     "id": "wh_123",
#     "url": "https://your-domain.com/webhook",
#     "events": ["job.completed", "job.failed"],
#     "status": "verified",
#     "active": true
#   }
# ]

# If status is "unverified", re-verify:
curl -X POST -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/webhooks/wh_123/verify"
```

### Common Webhook Problems & Fixes

| Problem                     | Cause                       | Solution                                           |
| --------------------------- | --------------------------- | -------------------------------------------------- |
| **"Connection refused"**    | Endpoint not accessible     | Ensure HTTPS, check firewall, allow AIPROD IPs     |
| **"Response timeout"**      | Endpoint too slow           | Optimize endpoint, return 202 Accepted immediately |
| **"SSL certificate error"** | Invalid/expired certificate | Renew certificate, ensure it matches domain        |
| **"404 Not Found"**         | Webhook path incorrect      | Verify URL path, redeploy endpoint                 |
| **"Signature mismatch"**    | Verification code wrong     | Use `X-AIPROD-Signature` header value              |

### Webhook Signature Verification

```python
# IMPORTANT: Always verify signatures to ensure requests are from AIPROD

import hmac
import hashlib
from flask import request, jsonify

WEBHOOK_SECRET = "whsec_..."  # From webhook registration

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    # Get signature from header
    signature = request.headers.get('X-AIPROD-Signature')

    # Get raw body
    body = request.get_data()

    # Compute expected signature
    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    # Verify
    if not hmac.compare_digest(signature, expected):
        return jsonify({"error": "Invalid signature"}), 401

    # Process webhook
    event = request.json
    print(f"Received event: {event['event']}")

    return jsonify({"received": True}), 200
```

---

## Rate Limiting & Quotas

### Understanding Rate Limits

**By Tier**:

```
Free Tier:
  - 100 jobs/month
  - 10 requests/second
  - 1 GB data/month

Pro Tier:
  - 10,000 jobs/month
  - 100 requests/second
  - 100 GB data/month

Enterprise:
  - Custom limits
  - Dedicated support
  - Custom contracts
```

### Issue: 429 Too Many Requests

**Symptoms**:

- Response status: `429 Too Many Requests`
- Headers contain: `Retry-After: 60` (wait 60 seconds)
- Error: `"Rate limit exceeded. Retry after 60 seconds"`

**Rate Limit Headers**:

```
X-RateLimit-Limit: 100        # Max requests per window
X-RateLimit-Remaining: 23     # Requests remaining
X-RateLimit-Reset: 1707039600 # Unix timestamp when limit resets
Retry-After: 23               # Seconds to wait before retry
```

**Handling 429 in Code**:

```python
# Python with exponential backoff
import time
import requests

def retry_with_backoff(url, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url)

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            continue

        return response

    raise Exception("Max retries exceeded")
```

```javascript
// JavaScript with exponential backoff
async function fetchWithRetry(url, maxRetries = 5) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const response = await fetch(url);

    if (response.status === 429) {
      const retryAfter = parseInt(response.headers.get("Retry-After")) || 60;
      console.log(`Rate limited. Waiting ${retryAfter}s...`);
      await new Promise((resolve) => setTimeout(resolve, retryAfter * 1000));
      continue;
    }

    return response;
  }

  throw new Error("Max retries exceeded");
}
```

### Quota Monitoring

```bash
# Check current usage
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/usage" | jq .

# Output:
# {
#   "jobs_this_month": 8500,
#   "jobs_limit": 10000,
#   "jobs_remaining": 1500,
#   "data_used_gb": 87,
#   "data_limit_gb": 100,
#   "rate_limit_remaining": 234,
#   "rate_limit_resets_at": "2026-03-04T00:00:00Z"
# }
```

### Optimization Strategies

**Batch Requests** (More efficient):

```python
# ❌ INEFFICIENT: 100 separate requests
for job_id in job_ids:
    response = requests.get(f"https://api.aiprod.com/jobs/{job_id}")
    # 100 requests = 100 against rate limit

# ✅ EFFICIENT: 1 batch request
response = requests.post(
    "https://api.aiprod.com/jobs/batch",
    json={"ids": job_ids}
)
# 1 request = minimal rate limit impact
```

**Caching** (Reduce redundant calls):

```python
import requests
from functools import lru_cache
import time

@lru_cache(maxsize=100)
def get_job_cached(job_id):
    return requests.get(f"https://api.aiprod.com/jobs/{job_id}").json()

# First call: API request
job = get_job_cached("job_123")

# Second call: Cached (no API request)
job = get_job_cached("job_123")

# Time-based cache invalidation (5 min)
def get_job_with_ttl(job_id, ttl=300):
    current_time = time.time()
    if current_time - get_job_with_ttl.last_reset > ttl:
        get_job_cached.cache_clear()
        get_job_with_ttl.last_reset = current_time
    return get_job_cached(job_id)

get_job_with_ttl.last_reset = time.time()
```

---

## Data Processing Issues

### Issue: Job Fails with "Invalid Input"

**Symptoms**:

- Job status: `failed`
- Error: `"Invalid input: URL returned HTTP 403"`
- Different file types fail inconsistently

**Debugging Input Issues**:

```bash
# Test your URL directly
curl -I https://your-file-url

# Check for:
# 1. Valid HTTP status (200, 206 for partial content)
# 2. Content-Type header
# 3. Content-Length header
# 4. No redirects (or max 2 redirects)

# Example output:
# HTTP/2 200
# Content-Type: audio/mpeg
# Content-Length: 12345678
# ✓ Acceptable
```

**Common Input Issues**:

| Error                          | Cause                        | Fix                                                |
| ------------------------------ | ---------------------------- | -------------------------------------------------- |
| **"HTTP 403 Forbidden"**       | File not publicly accessible | Make file public, add CORS headers                 |
| **"HTTP 404 Not Found"**       | URL invalid or deleted       | Verify URL is correct and active                   |
| **"HTTP 301 Redirect"**        | URL redirects (e.g., bit.ly) | Use final URL, or request supports redirect        |
| **"Unsupported Content-Type"** | File type not recognized     | Check MIME type, ensure file is valid format       |
| **"Timeout after 30s"**        | Server too slow              | Use CDN, optimize server, split into smaller files |

**Supported Input Formats**:

```
Video: MP4, WebM, AVI, MOV, MKV
  Max: 2GB per file, 10GB per job

Audio: MP3, WAV, OGG, FLAC, AAC
  Max: 500MB per file, 5GB per job

Images: JPEG, PNG, WebP, GIF
  Max: 100MB per file, 1GB per job

Documents: PDF, DOCX, TXT, JSON
  Max: 50MB per file, 500MB per job
```

---

### Issue: Job Timeout

**Symptoms**:

- Job status: `failed`
- Error: `"Timeout: Job exceeded 3600s limit"`
- No output generated

**Default Timeouts**:

```
Short jobs: 300s (5 min)      # Image processing, text extraction
Medium jobs: 1800s (30 min)   # Audio transcription, video conversion
Long jobs: 3600s (60 min)     # Large video processing, batch jobs
Custom: User specified, max 7200s
```

**Solutions**:

```python
# Option 1: Increase timeout
response = requests.post(
    "https://api.aiprod.com/jobs",
    json={
        "name": "Large video processing",
        "input_url": "https://...",
        "timeout": 7200  # 2 hours (max)
    }
)

# Option 2: Split large files
# Instead of: 1 x 5GB video → timeout
# Do: 5 x 1GB videos → process in parallel

# Option 3: Use streaming/chunking
# API supports streaming for large files:
with open('large_file.mp4', 'rb') as f:
    response = requests.post(
        "https://api.aiprod.com/jobs/stream",
        headers={"X-API-Key": "YOUR_KEY"},
        data=f,  # Stream body
        params={"filename": "large_file.mp4"}
    )
```

---

## Database Problems

### Issue: "Database Connection Failed"

**Symptoms**:

- Error appears in job logs
- Some jobs succeed, others fail randomly
- Correlates with high load periods

**Monitoring Database Health**:

```bash
# Check database status
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/diagnostics" | jq .database

# Expected healthy output:
# {
#   "status": "healthy",
#   "connection_pool_active": 12,
#   "connection_pool_available": 38,
#   "replication_lag_ms": 0,
#   "backup_age_hours": 0.5
# }

# Unhealthy signs:
# - status: "degraded" or "down"
# - connection_pool_available: 0 (pool exhausted)
# - replication_lag_ms > 1000 (replication delayed)
```

**Connection Pool Exhaustion**:

```python
# ❌ WRONG: Not releasing connections
import requests

response = requests.get("https://api.aiprod.com/jobs")
# Connection left open, accumulates with each request

# ✅ CORRECT: Use context manager
with requests.Session() as session:
    response = session.get("https://api.aiprod.com/jobs")
    # Connection released automatically
```

---

## Cloud Infrastructure Issues

### Issue: "Service Unavailable" (503)

**Symptoms**:

- Status: `503 Service Unavailable`
- Affects many users simultaneously
- Resolves after 15-60 minutes
- Check status page

**Typical Causes**:

| Cause                     | Duration         | Impact                    | Mitigation              |
| ------------------------- | ---------------- | ------------------------- | ----------------------- |
| **Planned Maintenance**   | 5-30 min         | All regions affected      | Check status.aiprod.com |
| **Database Upgrade**      | 15-60 min        | Some features unavailable | Use cached data         |
| **DDoS Attack**           | Minutes to hours | Regional impact           | Wait for mitigation     |
| **Cloud Provider Outage** | Hours            | Complete unavailability   | Incident in docs        |

**What To Do During Outage**:

```bash
# 1. Check status page
curl https://status.aiprod.com/api/status.json

# 2. Monitor service health
watch -n 5 'curl -s https://api.aiprod.com/health'

# 3. For critical operations, use cached results:
CACHE_FILE=last_successful_response.json
if [ -f "$CACHE_FILE" ]; then
    echo "Using cached response (outage)"
    cat "$CACHE_FILE"
fi
```

---

## Performance Issues

### Issue: Slow API Responses (>5s)

**Symptoms**:

- Normal requests taking 5-30 seconds
- Intermittent (not all requests affected)
- Correlates with time of day or job complexity

**Diagnosis**:

```bash
# 1. Check if your endpoint is responsible
curl -w '@curl-format.txt' -s https://api.aiprod.com/jobs/job_123

# curl-format.txt contents:
# time_namelookup: %{time_namelookup}
# time_connect: %{time_connect}
# time_appconnect: %{time_appconnect}
# time_starttransfer: %{time_starttransfer}
# time_total: %{time_total}

# Example output:
# time_namelookup: 0.002
# time_connect: 0.008
# time_appconnect: 0.025
# time_starttransfer: 0.120
# time_total: 0.143
```

**Performance Optimization**:

```python
# ❌ SLOW: Serial requests
import requests
jobs = []
for i in range(100):
    response = requests.get(f"https://api.aiprod.com/jobs/job_{i}")
    jobs.append(response.json())
# Takes 100 * 0.5s = 50 seconds

# ✅ FAST: Parallel requests
import requests
from concurrent.futures import ThreadPoolExecutor

def get_job(job_id):
    return requests.get(f"https://api.aiprod.com/jobs/{job_id}").json()

with ThreadPoolExecutor(max_workers=10) as executor:
    jobs = list(executor.map(get_job, [f"job_{i}" for i in range(100)]))
# Takes ~5 seconds (10x faster)
```

---

## Security & Access Issues

### Issue: IP Whitelist Blocking

**Symptoms**:

- You have IP whitelist enabled
- Connection refused from certain IPs
- Works from some locations, not others

**Managing IP Whitelist**:

```bash
# Get current whitelist
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/security" | jq .ip_whitelist

# Example output:
# {
#   "enabled": true,
#   "ips": [
#     "203.0.113.10/32",
#     "203.0.113.0/24"
#   ]
# }

# Add IP to whitelist (your current IP):
MYIP=$(curl -s https://icanhazip.com)
curl -X POST -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/security/whitelist" \
  -d "ip=$MYIP"

# Find your IP
curl https://icanhazip.com  # Returns: 203.0.113.10
```

---

## Integration Troubleshooting

### Issue: "Invalid Webhook Signature"

**Symptoms**:

- Webhook calls receiving `401 Unauthorized`
- Error: `"Invalid signature"`
- Other webhook endpoints work fine

**Signature Verification Checklist**:

```bash
# 1. Verify AIPROD is sending header
# (Enable request logging in your framework)

# Expected header from AIPROD:
X-AIPROD-Signature: sha256=abc123...

# 2. Verify you're using correct secret
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/webhooks/wh_123" | jq .secret

# 3. Verify signature algorithm
# AIPROD uses: HMAC-SHA256 (raw body, not JSON-encoded)
```

```python
# ❌ WRONG: Computing signature on JSON-encoded body
import hmac, hashlib, json
body = {"event": "job.completed"}
signature = hmac.new(b"secret", json.dumps(body).encode(), hashlib.sha256).hexdigest()

# ✅ CORRECT: Computing signature on raw body
import hmac, hashlib
raw_body = b'{"event": "job.completed"}'
signature = hmac.new(b"secret", raw_body, hashlib.sha256).hexdigest()
```

---

### Issue: Jobs Stuck in "Processing" State

**Symptoms**:

- Job status: `processing` for >24 hours
- No error in logs
- Job never moves to `completed` or `failed`

**Recovery Steps**:

```bash
# 1. Check job details
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/jobs/job_123" | jq .

# Look for:
# - last_activity_at (when was it last updated?)
# - worker_instance (which worker is processing it?)
# - attempted_count (how many retries?)

# 2. Force timeout (if stuck >24h)
curl -X POST -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/jobs/job_123/timeout" \
  -d '{"reason": "Stuck in processing"}'

# 3. Retry job
curl -X POST -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/jobs/job_123/retry"

# 4. If persistent, contact support with job ID
```

---

## FAQ & Advanced Diagnostics

### Q: How do I see actual API request/response for debugging?

**Answer**: Use `-v` flag with curl or network monitoring:

```bash
# curl verbose
curl -v https://api.aiprod.com/jobs/job_123

# Python with logging
import requests
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable HTTP request/response logging
import http.client as http_client
http_client.HTTPConnection.debuglevel = 1

requests.get("https://api.aiprod.com/jobs/job_123")
```

### Q: How do I report a bug or get emergency support?

**Answer**: Support levels by tier:

```
Free Tier:
  - Email only: support@aiprod.com
  - Response: 24-48 hours
  - No guaranteed SLA

Pro Tier:
  - Email + chat: support@aiprod.com
  - Response: 4 hours
  - Covered by SLA

Enterprise:
  - 24/7 phone + email
  - Dedicated account manager
  - 15-minute response time
```

**Emergency Contact (Production Down)**:

```bash
# For critical production outages:
# 1. Use web form: https://support.aiprod.com/emergency
# 2. Phone: +1-800-AIPROD-1 (Enterprise only)
# 3. Slack: #aiprod-support (if you have Enterprise Slack integration)
```

### Q: How can I test API before production?

**Answer**: Use sandbox environment:

```python
import requests

# Sandbox (free, safe for testing)
sandbox_url = "https://sandbox-api.aiprod.com"
sandbox_key = "sk_test_abc123"  # Test key

# Production (billable)
prod_url = "https://api.aiprod.com"
prod_key = "sk_live_xyz789"  # Live key

# Test in sandbox first
response = requests.get(
    f"{sandbox_url}/jobs",
    headers={"X-API-Key": sandbox_key}
)
```

### Q: How do I increase my rate limit?

**Answer**: Options by tier:

```
Free Tier:
  - Upgrade to Pro or Enterprise

Pro Tier:
  - Contact sales@aiprod.com
  - Provide usage justification
  - Typical: 100 → 500+ req/s

Enterprise:
  - Custom limits included
  - Negotiate in contract
```

---

## Troubleshooting Tools & Resources

### Useful Commands

```bash
# System information
curl -s -H "X-API-Key: YOUR_KEY" https://api.aiprod.com/diagnostics | jq .system

# Billing & usage
curl -s -H "X-API-Key: YOUR_KEY" https://api.aiprod.com/account/usage

# Recent errors (last 50)
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/logs?level=error&limit=50&sort=desc"

# Job metrics (last 7 days)
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://api.aiprod.com/account/metrics?period=7d" | jq .
```

### Debug Checklist

**Before Contacting Support**:

- [ ] Verified API key is correct and not revoked
- [ ] Confirmed request format using documentation examples
- [ ] Tested endpoint with curl to isolate client vs API issue
- [ ] Checked rate limit and quota usage
- [ ] Reviewed recent error logs in dashboard
- [ ] Confirmed input files are valid and accessible
- [ ] Tested in sandbox environment first
- [ ] Collected request ID (X-Request-ID header) from failed request

---

## Contact & Additional Support

**Documentation**: https://docs.aiprod.com  
**Status Page**: https://status.aiprod.com  
**Community Forum**: https://community.aiprod.com  
**Email Support**: support@aiprod.com  
**Enterprise Support**: enterprise@aiprod.com

---

**Version History**:

- v2.0 - February 4, 2026 - Comprehensive enhanced guide
- v1.0 - Initial version

---
