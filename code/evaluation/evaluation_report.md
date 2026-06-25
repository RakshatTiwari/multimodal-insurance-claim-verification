# Damage Claim Verification — Evaluation Report

## Accuracy on Sample Set (n=20)

| Field | Accuracy | Correct | Total |
|-------|----------|---------|-------|
| claim_status | 75.0% | 15 | 20 |
| issue_type | 30.0% | 6 | 20 |
| object_part | 80.0% | 16 | 20 |
| severity | 60.0% | 12 | 20 |
| evidence_standard_met | 90.0% | 18 | 20 |
| **Overall** | **70.8%** | **85** | **120** |

## Claim Status Breakdown (Precision / Recall / F1)

| Label | TP | FP | FN | Precision | Recall | F1 |
|-------|----|----|----|-----------|--------|----|
| contradicted | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 |
| not_enough_information | 2 | 1 | 0 | 0.67 | 1.00 | 0.80 |
| supported | 13 | 4 | 0 | 0.76 | 1.00 | 0.87 |

---

## Operational Analysis

### Sample Set Processing
- Claims processed: **20**
- Model calls: **20** (1 call per claim)
- Estimated input tokens: **51,600**
- Estimated output tokens: **6,000**
- Images processed: **28** (~1.4 avg per claim)
- Estimated cost: **$0.2448**
- Actual elapsed time: **7.0s**

### Test Set Estimates (n=44)
- Model calls: **44** (1 call per claim)
- Estimated input tokens: **113,520**
- Estimated output tokens: **13,200**
- Images processed: **61** (~1.4 avg per claim)
- Estimated cost: **$0.5386**
- Estimated latency: **176s** (~2.9 min)

---

## Rate Limits, Batching & Throttling Strategy

### TPM / RPM Considerations
- Each request sends 1–4 images + ~900 tokens of text context.
- A 1s inter-request delay is applied to stay well within Sonnet rate limits.

### Batching
- Claims are processed sequentially (1 model call per claim).
- Images for a single claim are combined into one request to minimize API calls.
- No parallel execution by default — add `--workers N` via `concurrent.futures` if throughput needs to increase.

### Caching
- No caching implemented currently.
- Potential improvement: cache model responses keyed by (image_hash, claim_hash) to avoid reprocessing identical inputs.

### Retry Strategy
- `MAX_RETRIES = 2` with exponential backoff on `RateLimitError`.
- Transient JSON parse errors also retry up to 2 times.
- Hard fallback row (all unknowns, `manual_review_required`) if all retries fail.

### Cost Optimisation Notes
- System prompt is kept short (~300 tokens) and reused across all calls.
- Images sent at original resolution; downscaling to 1024px would reduce image tokens by ~30%.
- For very large test sets, batching multiple claims in one prompt is possible but risks lower accuracy per claim.
