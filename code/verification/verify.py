"""
Damage Claim Verification System
Uses OpenRouter Vision Models to analyze images and verify insurance/damage claims.
"""

from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found in .env")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "X-Title": "Damage Claim Verification"
    }
)
import base64
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-nano-12b-v2-vl:free"
)

MAX_TOKENS = 512
MAX_RETRIES = 2
RETRY_DELAY = 2

VALID_CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}
VALID_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging", "water_damage",
    "stain", "none", "unknown",
}
VALID_SEVERITIES = {"none", "low", "medium", "high", "unknown"}
VALID_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

CAR_PARTS = {
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
    "body", "unknown",
}
LAPTOP_PARTS = {
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
    "port", "base", "body", "unknown",
}
PACKAGE_PARTS = {
    "box", "package_corner", "package_side", "seal", "label",
    "contents", "item", "unknown",
}
OBJECT_PARTS = {"car": CAR_PARTS, "laptop": LAPTOP_PARTS, "package": PACKAGE_PARTS}

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_evidence_requirements(path: str) -> dict:
    """Returns {(claim_object, applies_to): minimum_image_evidence}"""
    reqs = {}
    for row in load_csv(path):
        key = (row["claim_object"].strip(), row["applies_to"].strip())
        reqs[key] = row["minimum_image_evidence"].strip()
    return reqs


def load_user_history(path: str) -> dict:
    """Returns {user_id: row_dict}"""
    return {row["user_id"].strip(): row for row in load_csv(path)}


# ── Image helpers ─────────────────────────────────────────────────────────────

def get_image_id(image_path: str) -> str:
    return Path(image_path).stem


# ── Prompt builders ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert damage claim verification analyst. 
Your job is to analyze images and determine whether they support, contradict, 
or do not provide enough information to evaluate a damage claim.

You must return ONLY valid JSON.
Do not wrap the response in markdown.
Do not use ```json.
Do not explain your answer.

The JSON must contain exactly these fields:

{
  "evidence_standard_met": true | false,
  "evidence_standard_met_reason": "short reason string",
  "risk_flags": ["flag1", "flag2"],
  "issue_type": "one of the allowed values",
  "object_part": "one of the allowed values for this object type",
  "claim_status": "supported | contradicted | not_enough_information",
  "claim_status_justification": "concise image-grounded explanation",
  "supporting_image_ids": ["img_1", "img_2"],
  "valid_image": true | false,
  "severity": "none | low | medium | high | unknown"
}

Rules:
- Images are the PRIMARY source of truth.
- User history adds risk context only — it never overrides clear visual evidence.
- Use issue_type "none" when the part is visible but undamaged.
- Use "unknown" when the part or issue cannot be determined.
- supporting_image_ids should be the image IDs (filename without extension) 
  that support your decision. Use [] if none qualify.
- risk_flags must only include values from the allowed list.
"""


def build_user_message(
    claim_row,
    user_history,
    evidence_requirements,
    image_ids
):
    obj = claim_row["claim_object"].strip()

    allowed_parts = sorted(
        OBJECT_PARTS.get(obj, {"unknown"})
    )

    relevant_reqs = []

    for (req_obj, applies_to), rule in evidence_requirements.items():
        if req_obj in (obj, "all"):
            relevant_reqs.append(
                f"- {applies_to}: {rule}"
            )

    history_context = ""

    if user_history:
        history_context = f"""
USER HISTORY:
Past Claims: {user_history.get('past_claim_count')}
Rejected Claims: {user_history.get('rejected_claim')}
History Flags: {user_history.get('history_flags')}
Summary: {user_history.get('history_summary')}
"""

    text = f"""
{SYSTEM_PROMPT}

CLAIM OBJECT:
{obj}

USER CLAIM:
{claim_row['user_claim']}

IMAGE IDS:
{', '.join(image_ids)}

EVIDENCE REQUIREMENTS:
{chr(10).join(relevant_reqs)}

ALLOWED OBJECT PARTS:
{', '.join(allowed_parts)}

{history_context}

Return ONLY valid JSON.
"""

    return text


# ── API call with retry ───────────────────────────────────────────────────────

def image_to_data_url(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return f"data:image/jpeg;base64,{b64}"

def call_model(prompt_text, image_paths):

    last_error = None
    raw = None

    content = [{"type": "text", "text": prompt_text}]

    for path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(path)
            }
        })

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=MAX_TOKENS,
                timeout=60,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )

            raw = response.choices[0].message.content

            if raw is None:
                raise ValueError("OpenRouter returned empty content")

            raw = raw.strip()

            if raw.startswith("```"):
                raw = (
                    raw
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            try:
                return json.loads(raw)

            except Exception:

                start = raw.find("{")
                end = raw.rfind("}")

                if start >= 0 and end >= 0:
                    return json.loads(raw[start:end+1])

                raise

        except Exception as e:

            last_error = e

            print(
                f"Retry {attempt+1}/{MAX_RETRIES}: {e}",
                file=sys.stderr
            )

            time.sleep(RETRY_DELAY * (2 ** attempt))

    raise last_error


# ── Output sanitizer ──────────────────────────────────────────────────────────

def sanitize_output(result: dict, claim_row: dict, image_ids: list[str]) -> dict:
    """Ensure all output values are within allowed sets."""
    obj = claim_row["claim_object"].strip()

    # Claim status
    status = result.get("claim_status", "not_enough_information")
    if status not in VALID_CLAIM_STATUSES:
        status = "not_enough_information"

    # Issue type
    issue = result.get("issue_type", "unknown")
    if issue not in VALID_ISSUE_TYPES:
        issue = "unknown"

    # Object part
    allowed_parts = OBJECT_PARTS.get(obj, {"unknown"})
    part = result.get("object_part", "unknown")
    if part not in allowed_parts:
        part = "unknown"

    # Severity
    severity = result.get("severity", "unknown")
    if severity not in VALID_SEVERITIES:
        severity = "unknown"

    # Risk flags — filter to allowed set
    raw_flags = result.get("risk_flags", [])
    if isinstance(raw_flags, str):
        raw_flags = [f.strip() for f in raw_flags.split(";")]
    flags = [f for f in raw_flags if f in VALID_RISK_FLAGS]
    if not flags:
        flags = ["none"]

    # Supporting image IDs — ensure they're real image IDs
    sup_ids = result.get("supporting_image_ids", [])
    if isinstance(sup_ids, str):
        sup_ids = [s.strip() for s in sup_ids.split(";")]
    sup_ids = [i for i in sup_ids if i in image_ids]
    if not sup_ids:
        sup_ids = ["none"]

    return {
        "evidence_standard_met": str(result.get("evidence_standard_met", False)).lower(),
        "evidence_standard_met_reason": result.get("evidence_standard_met_reason", "").strip(),
        "risk_flags": ";".join(flags),
        "issue_type": issue,
        "object_part": part,
        "claim_status": status,
        "claim_status_justification": result.get("claim_status_justification", "").strip(),
        "supporting_image_ids": ";".join(sup_ids),
        "valid_image": str(result.get("valid_image", False)).lower(),
        "severity": severity,
    }


# ── Fallback for missing images ───────────────────────────────────────────────

def fallback_row(claim_row: dict, reason: str) -> dict:
    return {
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": reason,
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": f"Could not process images: {reason}",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


# ── Main processing loop ──────────────────────────────────────────────────────

def process_claims(
    claims_path: str,
    evidence_path: str,
    history_path: str,
    output_path: str,
    base_dir: str = ".",
    verbose: bool = True,
) -> list[dict]:
    claims = load_csv(claims_path)

    processed_claims = set()
    results = []

    if Path(output_path).exists():
        try:
            existing = load_csv(output_path)
            results = existing.copy()

            for row in existing:
                claim_key = (
                    row["user_id"],
                    row["image_paths"]
                )

                processed_claims.add(claim_key)

            print(
                f"Resuming from {len(processed_claims)} completed claims...",
                file=sys.stderr
            )

        except Exception:
            pass

    evidence_requirements = load_evidence_requirements(evidence_path)
    user_history_map = load_user_history(history_path)

    total = len(claims)

    for idx, claim_row in enumerate(claims, 1):
        user_id = claim_row["user_id"].strip()

        claim_key = (
            claim_row["user_id"],
            claim_row["image_paths"]
        )
        
        if claim_key in processed_claims:
            continue

        image_paths_raw = claim_row["image_paths"].strip()
        image_paths = [p.strip() for p in image_paths_raw.split(";") if p.strip()]
        image_ids = [get_image_id(p) for p in image_paths]

        if verbose:
            print(f"[{idx}/{total}] Processing claim for user {user_id} "
                  f"({len(image_paths)} image(s))...", file=sys.stderr)

        # Load images
        full_image_paths = []
        missing = []

        for path in image_paths:

            full_path = Path(base_dir) / path

            if not full_path.exists():
                full_path = Path(base_dir) / "dataset" / path

            if full_path.exists():
                full_image_paths.append(str(full_path))
            else:
                missing.append(path)

        if not full_image_paths:
            out = {**claim_row, **fallback_row(claim_row, f"No images found: {missing}")}
            results.append(out)

            # Save checkpoint after every claim
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=OUTPUT_COLUMNS,
                    extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(results)

            continue

        # Warn about missing but proceed with what we have
        if missing and verbose:
            print(f"  ⚠ Missing images: {missing}", file=sys.stderr)

        # Get user history
        user_hist = user_history_map.get(user_id)

        # Build message
        message_content = build_user_message(
            claim_row,
            user_hist,
            evidence_requirements,
            image_ids
        )

        # Call OpenRouter model
        try:
            raw_result = call_model(
                message_content,
                full_image_paths
            )

            if raw_result is None:
                raise ValueError("Model returned None")

            sanitized = sanitize_output(raw_result, claim_row, image_ids)

        except Exception as e:
            print(f"  ✗ Error on claim {user_id}: {e}", file=sys.stderr)
            sanitized = fallback_row(claim_row, f"API error: {str(e)[:100]}")

        out = {**claim_row, **sanitized}
        results.append(out)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=OUTPUT_COLUMNS,
                extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(results)

        # Brief throttle to avoid rate limits (TPM management)
        if idx < total:
            time.sleep(1)

    # Write output CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    if verbose:
        print(f"\n✓ Done. Output written to {output_path}", file=sys.stderr)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Damage Claim Verifier")
    parser.add_argument("--claims", default="dataset/claims.csv")
    parser.add_argument("--evidence", default="dataset/evidence_requirements.csv")
    parser.add_argument("--history", default="dataset/user_history.csv")
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--base-dir", default=".", help="Base directory for image paths")
    args = parser.parse_args()

    process_claims(
        claims_path=args.claims,
        evidence_path=args.evidence,
        history_path=args.history,
        output_path=args.output,
        base_dir=args.base_dir,
    )
