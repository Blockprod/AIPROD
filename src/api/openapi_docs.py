"""Advanced OpenAPI documentation with examples and schemas"""

from typing import Any, Dict, List
from enum import Enum


class PresetType(str, Enum):
    """Supported audio generation presets"""
    QUICK_SOCIAL = "quick_social"
    BRAND_CAMPAIGN = "brand_campaign"
    PREMIUM_SPOT = "premium_spot"


class PriorityLevel(str, Enum):
    """Job priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class LanguageCode(str, Enum):
    """Supported languages"""
    EN = "en"
    FR = "fr"
    ES = "es"
    DE = "de"
    IT = "it"
    PT = "pt"
    JA = "ja"
    ZH = "zh"
    KO = "ko"


# OpenAPI schema examples
PIPELINE_REQUEST_EXAMPLES = {
    "quick_social": {
        "description": "Quick social media post (15s)",
        "value": {
            "content": "Check out our new product! Amazing features, incredible price. Limited time offer!",
            "preset": "quick_social",
            "duration_sec": 15,
            "voice_id": "default",
            "lang": "en",
            "priority": "medium"
        }
    },
    "brand_campaign": {
        "description": "Full brand campaign spot (30s)",
        "value": {
            "content": "Welcome to the future of audio. Our cutting-edge technology brings your content to life with stunning voice synthesis, natural pacing, and emotional depth. Experience the difference.",
            "preset": "brand_campaign",
            "duration_sec": 30,
            "voice_id": "professional",
            "lang": "en",
            "priority": "high"
        }
    },
    "premium_spot": {
        "description": "Premium audio production (60s)",
        "value": {
            "content": "Introducing AudioSync Pro - the ultimate solution for content creators. With AI-powered voice synthesis, real-time collaboration, and industry-leading quality, AudioSync Pro sets the new standard. Available now with special launch pricing.",
            "preset": "premium_spot",
            "duration_sec": 60,
            "voice_id": "cinematic",
            "lang": "en",
            "priority": "high"
        }
    }
}

OPTIMIZE_REQUEST_EXAMPLES = {
    "basic": {
        "description": "Basic optimization request",
        "value": {
            "audio_url": "https://storage.googleapis.com/aiprod-bucket/sample.wav",
            "optimization_level": "balanced",
            "normalize_audio": True
        }
    },
    "aggressive": {
        "description": "Aggressive quality enhancement",
        "value": {
            "audio_url": "https://storage.googleapis.com/aiprod-bucket/sample.wav",
            "optimization_level": "aggressive",
            "normalize_audio": True,
            "reduce_noise": True,
            "enhance_clarity": True
        }
    }
}

JOB_STATUS_EXAMPLES = {
    "queued": {
        "description": "Job queued for processing",
        "value": {
            "job_id": "job-123456",
            "status": "queued",
            "progress": 0,
            "created_at": "2026-02-04T17:30:00Z",
            "estimated_completion": "2026-02-04T17:35:00Z"
        }
    },
    "processing": {
        "description": "Job currently processing",
        "value": {
            "job_id": "job-123456",
            "status": "processing",
            "progress": 45,
            "current_step": "voice_synthesis",
            "created_at": "2026-02-04T17:30:00Z",
            "started_at": "2026-02-04T17:31:00Z",
            "estimated_completion": "2026-02-04T17:34:00Z"
        }
    },
    "completed": {
        "description": "Job successfully completed",
        "value": {
            "job_id": "job-123456",
            "status": "completed",
            "progress": 100,
            "result": {
                "audio_url": "https://storage.googleapis.com/aiprod-bucket/output-123456.wav",
                "duration_sec": 15.5,
                "file_size_mb": 2.3,
                "format": "wav"
            },
            "created_at": "2026-02-04T17:30:00Z",
            "started_at": "2026-02-04T17:31:00Z",
            "completed_at": "2026-02-04T17:33:15Z",
            "processing_time_sec": 135
        }
    },
    "error": {
        "description": "Job failed with error",
        "value": {
            "job_id": "job-123456",
            "status": "error",
            "progress": 30,
            "error": {
                "code": "VOICE_SYNTHESIS_ERROR",
                "message": "Failed to synthesize voice: timeout during API call",
                "details": {"service": "suno-api", "retry_count": 3}
            },
            "created_at": "2026-02-04T17:30:00Z",
            "started_at": "2026-02-04T17:31:00Z",
            "failed_at": "2026-02-04T17:33:15Z"
        }
    }
}

# Response schemas
PIPELINE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string", "description": "Unique job identifier"},
        "status": {"type": "string", "enum": ["queued", "processing", "completed", "error"]},
        "message": {"type": "string", "description": "Status message"},
        "created_at": {"type": "string", "format": "date-time"},
        "estimated_completion": {"type": "string", "format": "date-time"}
    }
}

JOB_STATUS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string"},
        "status": {"type": "string", "enum": ["queued", "processing", "completed", "error"]},
        "progress": {"type": "integer", "minimum": 0, "maximum": 100},
        "current_step": {"type": "string"},
        "result": {
            "type": "object",
            "properties": {
                "audio_url": {"type": "string"},
                "duration_sec": {"type": "number"},
                "file_size_mb": {"type": "number"},
                "format": {"type": "string"}
            }
        },
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "details": {"type": "object"}
            }
        },
        "created_at": {"type": "string", "format": "date-time"},
        "started_at": {"type": "string", "format": "date-time"},
        "completed_at": {"type": "string", "format": "date-time"},
        "processing_time_sec": {"type": "number"}
    }
}

# OpenAPI tags
TAGS_METADATA = [
    {
        "name": "Health",
        "description": "System health and status endpoints"
    },
    {
        "name": "Pipeline",
        "description": "Audio generation pipeline endpoints"
    },
    {
        "name": "Jobs",
        "description": "Job management and monitoring"
    },
    {
        "name": "Optimization",
        "description": "Audio optimization and enhancement"
    },
    {
        "name": "Financial",
        "description": "Cost estimation and financial tracking"
    },
    {
        "name": "Admin",
        "description": "Administrative operations (requires authentication)"
    }
]

# API documentation sections
API_DOCUMENTATION = {
    "overview": """
# AIPROD API

Advanced AI-powered audio production platform for generating, optimizing, and managing audio content.

## Key Features
- Real-time voice synthesis with Suno AI
- Advanced audio optimization with specialized effects
- Batch processing for large-scale operations
- Cost estimation and financial tracking
- Comprehensive monitoring and logging

## Authentication
All endpoints require API key authentication via `X-API-Key` header.

## Rate Limiting
Different endpoints have different rate limits:
- Health endpoints: 1000 requests/minute
- Pipeline endpoints: 30 requests/minute
- Status endpoints: 60 requests/minute
- Optimization endpoints: 20 requests/minute

## Error Handling
All errors are returned as JSON with the following format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {}
  }
}
```
""",
    
    "pipeline_guide": """
# Audio Generation Pipeline

The pipeline processes content through multiple stages:

1. **Validation**: Content and parameters are validated
2. **Synthesis**: Voice synthesis using Suno AI
3. **Post-Processing**: Effects and enhancements applied
4. **Optimization**: Quality optimization and compression
5. **Storage**: Output saved to Cloud Storage
6. **Notification**: Completion notification via Pub/Sub

### Preset Types
- **quick_social**: 15-second optimized for social media
- **brand_campaign**: 30-second professional branding
- **premium_spot**: 60-second full production quality

### Example Workflow
1. Submit content via `/pipeline/run`
2. Receive `job_id` in response
3. Poll `/jobs/{job_id}/status` for progress
4. Retrieve result when status is `completed`
""",

    "batch_operations": """
# Batch Processing

Process multiple items simultaneously for better throughput.

## Batch Submission
Submit up to 100 items in a single request:
```json
{
  "items": [
    {"content": "First item..."},
    {"content": "Second item..."}
  ],
  "preset": "quick_social"
}
```

## Monitoring
- Poll `/batch/{batch_id}/status` for overall progress
- Retrieve individual job statuses
- Download results in bulk

## Best Practices
- Batch size: 10-50 items for optimal performance
- Maximum batch size: 100 items
- Recommended: Submit batches during off-peak hours
""",

    "error_codes": """
# Error Codes Reference

## Validation Errors (400)
- `INVALID_CONTENT`: Content format or length invalid
- `INVALID_PRESET`: Preset type not recognized
- `INVALID_DURATION`: Duration outside acceptable range

## Processing Errors (500)
- `VOICE_SYNTHESIS_ERROR`: Suno API failure
- `AUDIO_OPTIMIZATION_ERROR`: Post-processing failure
- `STORAGE_ERROR`: Could not save to Cloud Storage

## Rate Limit Errors (429)
- `RATE_LIMIT_EXCEEDED`: Too many requests
- Retry-After header indicates seconds until retry

## Authentication Errors (401/403)
- `INVALID_API_KEY`: API key not provided or invalid
- `INSUFFICIENT_PERMISSIONS`: API key lacks required scopes
"""
}


def get_endpoint_documentation(endpoint_name: str) -> Dict[str, Any]:
    """Get detailed documentation for specific endpoint"""
    
    documentation = {
        "pipeline_run": {
            "title": "Run Pipeline",
            "description": "Submit content for audio generation",
            "method": "POST",
            "path": "/pipeline/run",
            "tags": ["Pipeline"],
            "parameters": {
                "content": {
                    "type": "string",
                    "min_length": 10,
                    "max_length": 10000,
                    "description": "Content to synthesize"
                },
                "preset": {
                    "type": "string",
                    "enum": ["quick_social", "brand_campaign", "premium_spot"],
                    "description": "Generation preset"
                },
                "duration_sec": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 300,
                    "description": "Target duration in seconds"
                },
                "lang": {
                    "type": "string",
                    "enum": list(LanguageCode),
                    "description": "Language code"
                },
                "priority": {
                    "type": "string",
                    "enum": list(PriorityLevel),
                    "description": "Job priority level"
                }
            },
            "examples": PIPELINE_REQUEST_EXAMPLES,
            "responses": {
                "200": {
                    "description": "Job submitted successfully",
                    "schema": PIPELINE_RESPONSE_SCHEMA
                },
                "400": {"description": "Validation error"},
                "429": {"description": "Rate limit exceeded"}
            }
        },
        
        "job_status": {
            "title": "Get Job Status",
            "description": "Get status and progress of processing job",
            "method": "GET",
            "path": "/jobs/{job_id}/status",
            "tags": ["Jobs"],
            "parameters": {
                "job_id": {
                    "type": "string",
                    "description": "Job identifier",
                    "in": "path",
                    "required": True
                }
            },
            "examples": JOB_STATUS_EXAMPLES,
            "responses": {
                "200": {
                    "description": "Job status retrieved",
                    "schema": JOB_STATUS_RESPONSE_SCHEMA
                },
                "404": {"description": "Job not found"}
            }
        }
    }
    
    return documentation.get(endpoint_name, {})
