# API Integration Guide — AIPROD

**Version**: 1.0  
**Updated**: February 4, 2026  
**Status**: 🟢 Production Ready  
**Audience**: Partners, Developers, Integrators

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Making Requests](#making-requests)
4. [Handling Responses](#handling-responses)
5. [Webhooks](#webhooks)
6. [Rate Limiting](#rate-limiting)
7. [Pagination](#pagination)
8. [Error Handling](#error-handling)
9. [Code Examples](#code-examples)
10. [Best Practices](#best-practices)

---

## Getting Started

### Prerequisites

- Active AIPROD account (Free, Pro, or Enterprise tier)
- API Key from dashboard
- Basic understanding of REST APIs
- HTTP client (curl, Postman, SDK, or library)

### Quick Start (5 minutes)

#### Step 1: Get Your API Key

1. Log in: https://dashboard.aiprod.ai
2. Navigate: Settings → API Keys
3. Click: "Create New Key"
4. Select: Production environment
5. Copy: `sk_live_xxxxxxxxxxxxxxxx`

**Keep this secret!** Never commit to version control.

#### Step 2: Make Your First Request

```bash
curl -X POST https://api.aiprod.ai/pipeline/run \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/input.mp4",
    "voice_text": "Hello, this is a test",
    "voice_language": "en",
    "music_style": "orchestral"
  }'
```

**Expected Response (202 Accepted)**:

```json
{
  "job_id": "job_abc123def456",
  "status": "PENDING",
  "estimated_duration_seconds": 300,
  "created_at": "2026-02-04T10:30:00Z"
}
```

#### Step 3: Check Job Status

```bash
curl -X GET "https://api.aiprod.ai/pipeline/job_abc123def456" \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx"
```

**Response (200 OK)**:

```json
{
  "job_id": "job_abc123def456",
  "status": "COMPLETED",
  "result": {
    "video_url": "https://storage.googleapis.com/aiprod-results/job_abc123def456.mp4",
    "cost_estimate": 2.5,
    "quality_score": 0.92,
    "processing_time_seconds": 287
  }
}
```

---

## Authentication

### API Key Authentication

All requests must include your API key in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

### API Key Types

| Type         | Prefix     | Environment | Usage                 |
| ------------ | ---------- | ----------- | --------------------- |
| **Live Key** | `sk_live_` | Production  | Real jobs, real costs |
| **Test Key** | `sk_test_` | Sandbox     | Testing, development  |

### Environments

**Production Environment** (api.aiprod.ai)

- Real jobs processed
- Real costs charged
- Must use `sk_live_` keys

**Sandbox Environment** (sandbox-api.aiprod.ai)

- Simulated processing
- No charges
- Must use `sk_test_` keys
- Results returned instantly (test data)

### API Key Management

```bash
# Retrieve your API keys
curl https://api.aiprod.ai/keys \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx"

# Create new key
curl -X POST https://api.aiprod.ai/keys \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx" \
  -d '{"name": "My Mobile App"}'

# Revoke key
curl -X DELETE https://api.aiprod.ai/keys/KEY_ID \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx"

# Rotate key (create new, revoke old)
# Best practice: do monthly
```

### Security Best Practices

✅ **DO**:

- Store keys in environment variables
- Use different keys for different applications
- Rotate keys monthly
- Use keys with minimal required permissions
- Monitor API key usage

❌ **DON'T**:

- Hardcode keys in source code
- Share keys with other team members
- Use same key for multiple projects
- Log keys in error messages
- Commit keys to Git

---

## Making Requests

### Request Format

All requests must include:

```
Method:  POST / GET / PUT / DELETE
URL:     https://api.aiprod.ai/endpoint
Headers:
  - Authorization: Bearer YOUR_API_KEY
  - Content-Type: application/json (for POST/PUT)
Body:    JSON-encoded parameters
```

### Supported HTTP Methods

| Method     | Purpose            | Has Body |
| ---------- | ------------------ | -------- |
| **POST**   | Create resources   | Yes      |
| **GET**    | Retrieve resources | No       |
| **PUT**    | Update resources   | Yes      |
| **DELETE** | Delete resources   | No       |

### Common Endpoints

```
POST   /pipeline/run              # Submit job
GET    /pipeline/{job_id}         # Get job status
POST   /pipeline/batch            # Submit batch (Pro+)
GET    /jobs                      # List jobs
PUT    /jobs/{job_id}             # Update job
DELETE /jobs/{job_id}             # Cancel job (if pending)
GET    /presets                   # List presets
POST   /webhooks/register         # Register webhook
GET    /health                    # Health check
GET    /openapi.json              # OpenAPI schema
```

### Example: Submit Job

```bash
curl -X POST https://api.aiprod.ai/pipeline/run \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "voice_text": "Sample narration",
    "voice_language": "en",
    "music_style": "upbeat",
    "preset": "brand_campaign",
    "callback_url": "https://yourserver.com/callback",
    "metadata": {
      "customer_id": "cust_123",
      "project": "Q1_2026_Campaign"
    }
  }'
```

---

## Handling Responses

### Response Format

```json
{
  "status": "string",
  "data": {},
  "error": null,
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2026-02-04T10:30:00Z"
  }
}
```

### Status Codes

| Code    | Meaning      | Action                                  |
| ------- | ------------ | --------------------------------------- |
| **200** | OK           | Success, response body contains result  |
| **202** | Accepted     | Request accepted, processing async      |
| **400** | Bad Request  | Invalid request, check parameters       |
| **401** | Unauthorized | Invalid/missing API key                 |
| **403** | Forbidden    | Insufficient permissions                |
| **404** | Not Found    | Resource doesn't exist                  |
| **429** | Rate Limited | Too many requests, wait before retrying |
| **500** | Server Error | Temporary issue, retry after delay      |
| **503** | Unavailable  | Service down, retry later               |

### Successful Response (200)

```json
{
  "job_id": "job_123",
  "status": "COMPLETED",
  "result": {
    "video_url": "https://...",
    "cost_estimate": 2.5,
    "quality_score": 0.92
  }
}
```

### Async Response (202)

```json
{
  "job_id": "job_123",
  "status": "PENDING",
  "estimated_duration_seconds": 300
}
```

### Error Response (4xx/5xx)

```json
{
  "error": {
    "type": "validation_error",
    "message": "Invalid input",
    "details": {
      "voice_language": "Must be 2-letter code (e.g., 'en', 'fr')",
      "music_style": "Invalid style: 'invalid_style'"
    }
  }
}
```

---

## Webhooks

### Overview

Instead of polling for job completion, use webhooks for real-time notifications.

**Benefits**:

- No polling overhead (1 request vs. 10+)
- Real-time job completion notifications
- Automatic retries on failure
- Signature verification for security

### Register Webhook

```bash
curl -X POST https://api.aiprod.ai/webhooks/register \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://yourserver.com/aiprod-webhook",
    "events": ["job.completed", "job.failed"],
    "active": true
  }'
```

**Response**:

```json
{
  "webhook_id": "wh_abc123",
  "url": "https://yourserver.com/aiprod-webhook",
  "events": ["job.completed", "job.failed"],
  "secret": "whsec_xxxxxxxxxxxxxxxx"
}
```

**Store the `secret`** for signature verification.

### Webhook Events

| Event           | Trigger                         | Data                 |
| --------------- | ------------------------------- | -------------------- |
| `job.started`   | Job processing begins           | job_id, status       |
| `job.progress`  | Progress update (30%, 60%, 90%) | job_id, progress     |
| `job.completed` | Job finished successfully       | job_id, result, cost |
| `job.failed`    | Job processing failed           | job_id, error        |
| `job.cancelled` | Job was cancelled               | job_id, reason       |

### Webhook Payload

```json
{
  "event": "job.completed",
  "webhook_id": "wh_abc123",
  "data": {
    "job_id": "job_abc123",
    "status": "COMPLETED",
    "result": {
      "video_url": "https://...",
      "cost_estimate": 2.5,
      "quality_score": 0.92
    }
  },
  "timestamp": "2026-02-04T10:30:00Z"
}
```

### Verify Webhook Signature

```python
import hmac
import hashlib
import json

def verify_webhook(payload_bytes, signature, secret):
    """Verify webhook signature"""
    expected_sig = hmac.new(
        secret.encode(),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(signature, expected_sig)

# Usage in Flask
from flask import request

@app.route('/aiprod-webhook', methods=['POST'])
def handle_webhook():
    payload = request.get_data()
    signature = request.headers.get('X-AIPROD-Signature')

    if not verify_webhook(payload, signature, WEBHOOK_SECRET):
        return {'error': 'Invalid signature'}, 401

    data = json.loads(payload)
    # Process webhook...

    return {'ok': True}
```

### Webhook Retry Policy

- **First retry**: 1 second
- **Second retry**: 5 seconds
- **Third retry**: 30 seconds
- **Max attempts**: 5 times over 24 hours
- **Timeout**: 30 seconds per attempt

### Webhook Management

```bash
# List webhooks
curl https://api.aiprod.ai/webhooks \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx"

# Update webhook
curl -X PUT https://api.aiprod.ai/webhooks/wh_abc123 \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx" \
  -d '{"active": false}'

# Delete webhook
curl -X DELETE https://api.aiprod.ai/webhooks/wh_abc123 \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx"

# Test webhook delivery
curl -X POST https://api.aiprod.ai/webhooks/wh_abc123/test \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx"
```

---

## Rate Limiting

### Rate Limits by Tier

| Tier       | Requests/Minute | Concurrent Jobs | Burst       |
| ---------- | --------------- | --------------- | ----------- |
| Free       | 10              | 2               | 5 requests  |
| Pro        | 100             | 10              | 50 requests |
| Enterprise | Unlimited       | 50              | Unlimited   |

### Rate Limit Headers

Every response includes:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1675338600
```

### When Rate Limited (429)

Response:

```json
{
  "error": {
    "type": "rate_limit_error",
    "message": "Rate limit exceeded",
    "retry_after": 60
  }
}
```

### Handling Rate Limits

```python
import time
import requests

def make_request_with_retry(url, headers, data=None):
    """Make request with exponential backoff"""

    for attempt in range(5):
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 429:
            # Extract retry-after header
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited, waiting {retry_after}s...")
            time.sleep(retry_after)
            continue

        return response

    raise Exception("Failed after 5 attempts")
```

### Upgrade Tier to Increase Limits

Go to https://dashboard.aiprod.ai/settings/tier and upgrade to Pro or Enterprise.

---

## Pagination

### List Endpoints

Endpoints that return multiple items support pagination:

```
GET /jobs                   # List all jobs
GET /presets                # List all presets
GET /webhooks              # List all webhooks
```

### Query Parameters

| Parameter | Type    | Default     | Max |
| --------- | ------- | ----------- | --- |
| `page`    | integer | 1           | -   |
| `limit`   | integer | 10          | 100 |
| `sort`    | string  | -created_at | -   |

### Pagination Response

```json
{
  "items": [
    {"job_id": "job_1", ...},
    {"job_id": "job_2", ...}
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 145,
    "pages": 15,
    "has_next": true,
    "has_previous": false
  }
}
```

### Example: Fetch All Jobs

```bash
# Page 1
curl "https://api.aiprod.ai/jobs?page=1&limit=50" \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx"

# Page 2
curl "https://api.aiprod.ai/jobs?page=2&limit=50" \
  -H "Authorization: Bearer sk_live_xxxxxxxxxxxxxxxx"

# Continue until has_next = false
```

### Sorting

```bash
# Oldest first
curl "https://api.aiprod.ai/jobs?sort=created_at" ...

# Newest first (default)
curl "https://api.aiprod.ai/jobs?sort=-created_at" ...

# By status
curl "https://api.aiprod.ai/jobs?sort=status" ...
```

---

## Error Handling

### Error Response Structure

```json
{
  "error": {
    "type": "error_type",
    "message": "Human-readable message",
    "code": "ERROR_CODE",
    "details": {
      "field_name": "Field-specific error message"
    }
  }
}
```

### Common Errors

#### 400 - Bad Request

**Cause**: Invalid input parameter

```json
{
  "error": {
    "type": "validation_error",
    "message": "Invalid input",
    "details": {
      "voice_language": "Must be 2-letter code (e.g., 'en', 'fr')"
    }
  }
}
```

**Fix**: Validate input before sending

#### 401 - Unauthorized

**Cause**: Invalid or missing API key

```json
{
  "error": {
    "type": "auth_error",
    "message": "Invalid API key"
  }
}
```

**Fix**: Check your API key

#### 403 - Forbidden

**Cause**: Insufficient permissions for operation

```json
{
  "error": {
    "type": "permission_error",
    "message": "Your tier does not support batch processing"
  }
}
```

**Fix**: Upgrade tier or request feature access

#### 404 - Not Found

**Cause**: Resource doesn't exist

```json
{
  "error": {
    "type": "not_found_error",
    "message": "Job not found"
  }
}
```

**Fix**: Verify job_id is correct

#### 429 - Rate Limited

See [Rate Limiting](#rate-limiting) section

#### 500 - Server Error

**Cause**: Internal server error (temporary)

```json
{
  "error": {
    "type": "server_error",
    "message": "Internal server error"
  }
}
```

**Fix**: Retry after waiting

### Error Handling Pattern

```python
import requests
import time

def handle_api_request(endpoint, data=None, max_retries=3):
    """Generic API request with error handling"""

    url = f"https://api.aiprod.ai/{endpoint}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    for attempt in range(max_retries):
        try:
            if data:
                response = requests.post(url, json=data, headers=headers)
            else:
                response = requests.get(url, headers=headers)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            # Handle client errors
            if 400 <= response.status_code < 500:
                error = response.json()
                print(f"Client error: {error['error']['message']}")
                raise ValueError(error)

            # Handle server errors with retry
            if 500 <= response.status_code < 600:
                print(f"Server error, attempt {attempt + 1}/{max_retries}")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue

            # Success
            return response.json()

        except requests.RequestException as e:
            print(f"Network error: {e}")
            time.sleep(2 ** attempt)
            continue

    raise Exception(f"Failed after {max_retries} attempts")
```

---

## Code Examples

### Python (requests)

```python
import requests
import json

API_KEY = "sk_live_xxxxxxxxxxxxxxxx"
BASE_URL = "https://api.aiprod.ai"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Submit job
response = requests.post(
    f"{BASE_URL}/pipeline/run",
    headers=headers,
    json={
        "video_url": "https://example.com/video.mp4",
        "voice_text": "Hello world",
        "voice_language": "en",
        "music_style": "orchestral",
        "preset": "brand_campaign"
    }
)

job_data = response.json()
job_id = job_data["job_id"]
print(f"Job submitted: {job_id}")

# Poll for completion
import time

while True:
    response = requests.get(
        f"{BASE_URL}/pipeline/{job_id}",
        headers=headers
    )

    job = response.json()
    print(f"Status: {job['status']}")

    if job['status'] == 'COMPLETED':
        print(f"Result: {job['result']['video_url']}")
        break
    elif job['status'] == 'FAILED':
        print(f"Error: {job['error']}")
        break

    time.sleep(10)
```

### Node.js (fetch)

```javascript
const API_KEY = "sk_live_xxxxxxxxxxxxxxxx";
const BASE_URL = "https://api.aiprod.ai";

async function submitJob() {
  const response = await fetch(`${BASE_URL}/pipeline/run`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      video_url: "https://example.com/video.mp4",
      voice_text: "Hello world",
      voice_language: "en",
      music_style: "orchestral",
    }),
  });

  const job = await response.json();
  console.log(`Job submitted: ${job.job_id}`);

  return job.job_id;
}

async function checkJobStatus(jobId) {
  const response = await fetch(`${BASE_URL}/pipeline/${jobId}`, {
    headers: { Authorization: `Bearer ${API_KEY}` },
  });

  return await response.json();
}

// Usage
const jobId = await submitJob();

let completed = false;
while (!completed) {
  const job = await checkJobStatus(jobId);
  console.log(`Status: ${job.status}`);

  if (job.status === "COMPLETED") {
    console.log(`Video: ${job.result.video_url}`);
    completed = true;
  } else if (job.status === "FAILED") {
    console.error(`Error: ${job.error}`);
    completed = true;
  }

  // Wait before polling again
  await new Promise((resolve) => setTimeout(resolve, 10000));
}
```

### cURL

```bash
#!/bin/bash

API_KEY="sk_live_xxxxxxxxxxxxxxxx"
BASE_URL="https://api.aiprod.ai"

# Submit job
JOB_RESPONSE=$(curl -s -X POST "$BASE_URL/pipeline/run" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "voice_text": "Hello world",
    "voice_language": "en",
    "music_style": "orchestral"
  }')

JOB_ID=$(echo $JOB_RESPONSE | jq -r '.job_id')
echo "Job submitted: $JOB_ID"

# Poll for completion
while true; do
  JOB=$(curl -s -X GET "$BASE_URL/pipeline/$JOB_ID" \
    -H "Authorization: Bearer $API_KEY")

  STATUS=$(echo $JOB | jq -r '.status')
  echo "Status: $STATUS"

  if [ "$STATUS" = "COMPLETED" ]; then
    VIDEO_URL=$(echo $JOB | jq -r '.result.video_url')
    echo "Video: $VIDEO_URL"
    break
  elif [ "$STATUS" = "FAILED" ]; then
    ERROR=$(echo $JOB | jq -r '.error')
    echo "Error: $ERROR"
    break
  fi

  sleep 10
done
```

---

## Best Practices

### 1. Use Webhooks Instead of Polling

❌ **Inefficient (Polling)**:

```python
while True:
    response = requests.get(f"/jobs/{job_id}")
    if response['status'] == 'COMPLETED':
        break
    time.sleep(5)  # 10+ requests per job!
```

✅ **Efficient (Webhooks)**:

```python
# Register once
requests.post('/webhooks/register', json={
    'url': 'https://yourserver.com/webhook',
    'events': ['job.completed']
})

# Receive notification when done (1 request per job)
```

### 2. Implement Exponential Backoff

❌ **Fixed delay**:

```python
time.sleep(5)  # Always wait 5s
```

✅ **Exponential backoff**:

```python
for attempt in range(5):
    try:
        response = api_call()
        return response
    except Exception:
        time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s, 16s
```

### 3. Cache API Responses

❌ **Repeated requests**:

```python
# Call every time user refreshes page
presets = requests.get('/presets').json()
```

✅ **Cache results**:

```python
# Cache for 1 hour
cache.set('presets', presets, 3600)
presets = cache.get('presets')  # Instant
```

### 4. Batch Operations When Possible

❌ **Individual requests** (10+ API calls):

```python
for video in videos:
    requests.post('/pipeline/run', json={'video_url': video})
```

✅ **Batch request** (1 API call):

```python
requests.post('/pipeline/batch', json={
    'videos': [{'url': v} for v in videos]
})
```

### 5. Handle Errors Gracefully

❌ **No error handling**:

```python
response = requests.get(url)
data = response.json()  # Crashes if error
```

✅ **Proper error handling**:

```python
response = requests.get(url)
if response.status_code != 200:
    print(f"Error: {response.json()['error']}")
else:
    data = response.json()
```

### 6. Monitor API Usage

```python
# Log every API call
import logging

logger = logging.getLogger('api')

@app.before_request
def log_request():
    logger.info(f"{request.method} {request.path}")

@app.after_request
def log_response(response):
    logger.info(f"Response: {response.status_code}")
    return response

# Periodically check usage
requests.get('/me/usage').json()
```

### 7. Rotate API Keys Regularly

```bash
# Monthly rotation
1. Create new key
2. Update applications with new key
3. Test thoroughly
4. Revoke old key

# Audit key usage
curl https://api.aiprod.ai/audit/keys \
  -H "Authorization: Bearer sk_live_..."
```

---

## Troubleshooting

### Common Issues

**Issue**: "Invalid API key"

- Check key prefix (`sk_live_` or `sk_test_`)
- Verify key hasn't been revoked
- Ensure header format: `Bearer sk_live_...`

**Issue**: "Rate limited"

- See [Rate Limiting](#rate-limiting) section
- Implement exponential backoff
- Upgrade tier for higher limits

**Issue**: Job stuck in PENDING

- Check Cloud Logging for errors
- Verify input parameters valid
- Use webhook for notifications instead of polling

**Issue**: "Permission denied" (403)

- Check your tier (batch requires Pro+)
- Verify API key has necessary scopes
- Contact support for scope issues

---

## Support

- **Documentation**: https://docs.aiprod.ai
- **Status Page**: https://status.aiprod.ai
- **Email**: support@aiprod.ai
- **Chat** (Pro+): https://dashboard.aiprod.ai/chat
- **Phone** (Enterprise): +1-XXX-XXX-XXXX

---

**Document Status**: 🟢 Production Ready  
**Last Updated**: February 4, 2026  
**Version**: 1.0  
**SDK Support**: Python, Node.js, Java, Go (coming soon)
