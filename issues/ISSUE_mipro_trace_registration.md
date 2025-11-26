# MIPRO Trace Registration Failure with Cloudflare Tunnels

## Summary
MIPRO jobs running through Cloudflare tunnels fail mid-bootstrap with "No trace registration found for correlation_id" error.

## Reproduction
- Job ID: `pl_2a6f86209a4f407c`
- Setup: MIPRO job with task app behind Cloudflare tunnel
- Error occurs during bootstrap phase after stage config is accepted

## Error Message
```
CRITICAL: No trace registration found for correlation_id=...
```

## Expected Behavior
Trace pipeline should register correlation IDs for jobs regardless of whether the task app is accessed via tunnel or direct connection.

## Root Cause Hypothesis
The trace registration happens via a different path than the main job execution, and the tunnel URL is not being properly associated with the trace pipeline's correlation tracking.

Possible issues:
1. Trace registration uses a different URL than the job's task_app_url
2. Correlation ID is generated but not stored before bootstrap requests arrive
3. Race condition between job submission and trace pipeline initialization

## Investigation Steps
1. Check trace pipeline registration logic in backend
2. Verify correlation ID lifecycle (generation → storage → lookup)
3. Check if tunnel URLs are handled differently than direct URLs in trace registration

## Labels
`bug`, `mipro`, `trace-pipeline`, `cloudflare-tunnel`

