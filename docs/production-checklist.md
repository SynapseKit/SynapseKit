# Production checklist

Use this one-page checklist before shipping a SynapseKit app to production.

## Observability
- [ ] Tracing enabled for LLM + RAG + agents (see docs/observability.md)
- [ ] Metrics enabled (`synapsekit_cost_usd_total`, `synapsekit_tokens_total`, `synapsekit_latency_seconds`)
- [ ] Log redaction in place for secrets and PII
- [ ] Alerts configured for error rate and latency spikes

## Eval gates
- [ ] Eval suite selected (EvalHub or internal suite)
- [ ] Minimum quality thresholds defined per task
- [ ] Regression checks run on prompt/model changes
- [ ] Fallback model/provider defined for failed eval gates

## Rate limiting & retries
- [ ] Per-user and global rate limits configured
- [ ] Retries with exponential backoff and jitter
- [ ] Timeouts set for all external calls
- [ ] Circuit breaker or bulkhead strategy for provider outages

## Secrets & access
- [ ] API keys stored in a secret manager (not in code or .env committed)
- [ ] Rotation policy defined and tested
- [ ] Least-privilege scopes for all provider keys
- [ ] Audit logging for key access/changes

## Cost controls
- [ ] Budget alerts configured per provider/project
- [ ] Cost tracking enabled (observability + billing dashboard)
- [ ] Token/latency caps enforced per request
- [ ] Batch/streaming mode selected for cost efficiency

## Failure modes
- [ ] Degraded mode behavior documented (reduced features / safe defaults)
- [ ] Fallback responses for tool/LLM failures
- [ ] Queueing strategy for burst traffic
- [ ] Data loss prevention for in-flight requests

## Pre‑prod sign‑off
- [ ] Owner: ____________________
- [ ] Environment: ______________
- [ ] Date: _____________________
