# SFT File Upload 503 - Wasabi S3 Connectivity from Render

## Summary
SFT dataset uploads to `agent-learning.onrender.com/api/files` fail with 503 error due to Wasabi S3 connectivity issues.

## Reproduction
```bash
# Upload training file via SDK
await sft_client.upload_training_file("training_data.jsonl")
```

## Error
```
HTTP 503: file_metadata_write_failed
Underlying: No route to host (s3.wasabisys.com)
```

## Environment
- Backend: `agent-learning.onrender.com` (Render deployment)
- Storage: Wasabi S3 (`s3.wasabisys.com`)

## Root Cause
Render backend cannot establish network connection to Wasabi S3 endpoints. This is likely:
1. Firewall/egress rules on Render blocking S3 traffic
2. DNS resolution failure for Wasabi endpoints
3. Wasabi service outage or IP block

## Investigation Steps
1. Check Render service logs for network errors
2. Verify Wasabi credentials are correct in Render env vars
3. Test S3 connectivity from Render shell:
   ```bash
   curl -I https://s3.wasabisys.com
   ```
4. Check if Render has egress IP restrictions that Wasabi might be blocking

## Workaround
For local development, set `TESTING=true` to bypass Wasabi validation.
Production fix requires resolving network connectivity.

## Labels
`bug`, `infra`, `sft`, `wasabi`, `render`

