"""
Evaluation script: compares system predictions against sample_claims.csv ground truth.
Produces evaluation/evaluation_report.md and evaluation/eval_metrics.json.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "verification"))
from verify import process_claims, load_csv, OUTPUT_COLUMNS

EVAL_COLUMNS = ["claim_status", "issue_type", "object_part", "severity",
                "evidence_standard_met", "valid_image"]

REPORT_PATH = Path(__file__).parent / "evaluation_report.md"
METRICS_PATH = Path(__file__).parent / "eval_metrics.json"
PRED_PATH = Path(__file__).parent / "eval_predictions.csv"


def score_predictions(ground_truth: list[dict], predictions: list[dict]) -> dict:
    """Compute per-column accuracy and overall metrics."""
    if len(ground_truth) != len(predictions):
        print(f"Warning: GT rows={len(ground_truth)}, Pred rows={len(predictions)}")

    n = min(len(ground_truth), len(predictions))
    column_correct = {col: 0 for col in EVAL_COLUMNS}
    column_total = {col: 0 for col in EVAL_COLUMNS}
    per_row = []

    for i in range(n):
        gt = ground_truth[i]
        pred = predictions[i]
        row_scores = {}
        for col in EVAL_COLUMNS:
            gt_val = str(gt.get(col, "")).strip().lower()
            pred_val = str(pred.get(col, "")).strip().lower()
            if gt_val:  # only score where GT has a value
                match = gt_val == pred_val
                row_scores[col] = {"gt": gt_val, "pred": pred_val, "match": match}
                column_correct[col] += int(match)
                column_total[col] += 1
        per_row.append(row_scores)

    metrics = {}
    for col in EVAL_COLUMNS:
        total = column_total[col]
        correct = column_correct[col]
        metrics[col] = {
            "accuracy": round(correct / total, 4) if total else None,
            "correct": correct,
            "total": total,
        }

    # Overall accuracy across all scored fields
    all_correct = sum(column_correct.values())
    all_total = sum(column_total.values())
    metrics["overall"] = {
        "accuracy": round(all_correct / all_total, 4) if all_total else None,
        "correct": all_correct,
        "total": all_total,
    }

    # Claim status breakdown
    status_gt = [gt.get("claim_status", "").strip().lower() for gt in ground_truth[:n]]
    status_pred = [pred.get("claim_status", "").strip().lower() for pred in predictions[:n]]
    status_labels = sorted(set(status_gt))
    confusion = {label: {"tp": 0, "fp": 0, "fn": 0} for label in status_labels}
    for gt_val, pred_val in zip(status_gt, status_pred):
        if gt_val in confusion:
            if gt_val == pred_val:
                confusion[gt_val]["tp"] += 1
            else:
                confusion[gt_val]["fn"] += 1
        if pred_val in confusion and pred_val != gt_val:
            confusion[pred_val]["fp"] += 1

    metrics["claim_status_breakdown"] = confusion
    metrics["per_row_scores"] = per_row

    return metrics


def estimate_costs(n_claims: int, n_images_avg: float = 1.5) -> dict:
    """Rough cost / token / latency estimates."""
    input_price_per_mtok = 3.00   # $3 per million input tokens
    output_price_per_mtok = 15.00  # $15 per million output tokens

    # Rough token estimates per claim
    system_tokens = 300
    text_context_tokens = 600
    image_tokens_each = 1200  # ~1200 tokens per typical image at medium res
    output_tokens = 300

    input_per_claim = system_tokens + text_context_tokens + (image_tokens_each * n_images_avg)
    output_per_claim = output_tokens

    total_input = input_per_claim * n_claims
    total_output = output_per_claim * n_claims

    cost = (total_input / 1_000_000 * input_price_per_mtok +
            total_output / 1_000_000 * output_price_per_mtok)

    latency_per_claim = 3.5  # seconds average
    total_latency = latency_per_claim * n_claims + (n_claims * 0.5)  # +throttle

    return {
        "n_claims": n_claims,
        "avg_images_per_claim": n_images_avg,
        "total_images": int(n_claims * n_images_avg),
        "total_input_tokens_est": int(total_input),
        "total_output_tokens_est": int(total_output),
        "cost_usd_est": round(cost, 4),
        "total_latency_seconds_est": round(total_latency, 1),
        "pricing_assumptions": {
            "model": "nvidia/nemotron-nano-12b-v2-vl:free",
            "input_per_mtok_usd": input_price_per_mtok,
            "output_per_mtok_usd": output_price_per_mtok,
            "tokens_per_image": image_tokens_each,
        },
    }


def write_report(
    metrics: dict,
    sample_cost: dict,
    test_cost: dict,
    n_sample: int,
    n_test: int,
    elapsed: float,
) -> str:
    cs = metrics.get("claim_status", {})
    it = metrics.get("issue_type", {})
    op = metrics.get("object_part", {})
    sv = metrics.get("severity", {})
    ov = metrics.get("overall", {})
    esm = metrics.get("evidence_standard_met", {})

    def pct(m):
        a = m.get("accuracy")
        return f"{a*100:.1f}%" if a is not None else "N/A"

    breakdown = metrics.get("claim_status_breakdown", {})
    breakdown_lines = []
    for label, counts in breakdown.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        breakdown_lines.append(
            f"| {label} | {tp} | {fp} | {fn} | {prec:.2f} | {rec:.2f} | {f1:.2f} |"
        )

    report = f"""# Damage Claim Verification — Evaluation Report

## Accuracy on Sample Set (n={n_sample})

| Field | Accuracy | Correct | Total |
|-------|----------|---------|-------|
| claim_status | {pct(cs)} | {cs.get('correct','?')} | {cs.get('total','?')} |
| issue_type | {pct(it)} | {it.get('correct','?')} | {it.get('total','?')} |
| object_part | {pct(op)} | {op.get('correct','?')} | {op.get('total','?')} |
| severity | {pct(sv)} | {sv.get('correct','?')} | {sv.get('total','?')} |
| evidence_standard_met | {pct(esm)} | {esm.get('correct','?')} | {esm.get('total','?')} |
| **Overall** | **{pct(ov)}** | **{ov.get('correct','?')}** | **{ov.get('total','?')}** |

## Claim Status Breakdown (Precision / Recall / F1)

| Label | TP | FP | FN | Precision | Recall | F1 |
|-------|----|----|----|-----------|--------|----|
{chr(10).join(breakdown_lines)}

---

## Operational Analysis

### Sample Set Processing
- Claims processed: **{n_sample}**
- Model calls: **{n_sample}** (1 call per claim)
- Estimated input tokens: **{sample_cost['total_input_tokens_est']:,}**
- Estimated output tokens: **{sample_cost['total_output_tokens_est']:,}**
- Images processed: **{sample_cost['total_images']}** (~{sample_cost['avg_images_per_claim']} avg per claim)
- Estimated cost: **${sample_cost['cost_usd_est']:.4f}**
- Actual elapsed time: **{elapsed:.1f}s**

### Test Set Estimates (n={n_test})
- Model calls: **{n_test}** (1 call per claim)
- Estimated input tokens: **{test_cost['total_input_tokens_est']:,}**
- Estimated output tokens: **{test_cost['total_output_tokens_est']:,}**
- Images processed: **{test_cost['total_images']}** (~{test_cost['avg_images_per_claim']} avg per claim)
- Estimated cost: **${test_cost['cost_usd_est']:.4f}**
- Estimated latency: **{test_cost['total_latency_seconds_est']:.0f}s** (~{test_cost['total_latency_seconds_est']/60:.1f} min)

### Pricing Assumptions
- Model: `{sample_cost['pricing_assumptions']['model']}`
- Input: ${sample_cost['pricing_assumptions']['input_per_mtok_usd']}/M tokens
- Output: ${sample_cost['pricing_assumptions']['output_per_mtok_usd']}/M tokens
- ~{sample_cost['pricing_assumptions']['tokens_per_image']} tokens per image (medium resolution estimate)

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
"""
    return report


def main():
    base_dir = Path(__file__).parent.parent.parent
    sample_path = base_dir / "dataset" / "sample_claims.csv"
    claims_path = base_dir / "dataset" / "claims.csv"
    evidence_path = base_dir / "dataset" / "evidence_requirements.csv"
    history_path = base_dir / "dataset" / "user_history.csv"
    eval_dir = base_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    if not sample_path.exists():
        print(f"Error: {sample_path} not found. Place your dataset files in dataset/", file=sys.stderr)
        sys.exit(1)

    sample_gt = load_csv(str(sample_path))
    n_sample = len(sample_gt)
    print(f"Running evaluation on {n_sample} sample claims...", file=sys.stderr)

    t0 = time.time()
    predictions = process_claims(
        claims_path=str(sample_path),
        evidence_path=str(evidence_path),
        history_path=str(history_path),
        output_path=str(PRED_PATH),
        base_dir=str(base_dir),
        verbose=True,
    )
    elapsed = time.time() - t0

    metrics = score_predictions(sample_gt, predictions)

    # Save metrics JSON
    metrics_out = {k: v for k, v in metrics.items() if k != "per_row_scores"}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Count test claims if available
    n_test = len(load_csv(str(claims_path))) if claims_path.exists() else n_sample * 5

    # Estimate avg images
    def avg_images(rows):
        counts = [len(r["image_paths"].split(";")) for r in rows if r.get("image_paths")]
        return round(sum(counts) / len(counts), 1) if counts else 1.5

    sample_avg_img = avg_images(sample_gt)
    sample_cost = estimate_costs(n_sample, sample_avg_img)
    test_cost = estimate_costs(n_test, sample_avg_img)

    report = write_report(metrics, sample_cost, test_cost, n_sample, n_test, elapsed)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Evaluation complete!", file=sys.stderr)
    print(f"  Overall accuracy: {metrics['overall'].get('accuracy', 0)*100:.1f}%", file=sys.stderr)
    print(f"  Claim status accuracy: {metrics['claim_status'].get('accuracy', 0)*100:.1f}%", file=sys.stderr)
    print(f"  Report: {REPORT_PATH}", file=sys.stderr)
    print(f"  Metrics: {METRICS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
