"""
extraction.py — LLM extraction of PA policy parameters + access score
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from groq import Groq


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT_TEMPLATE = """
You are a pharmaceutical market access analyst extracting Prior Authorization (PA) policy parameters.
Return ONLY valid JSON. No markdown, no prose, no explanation.

━━━━━━━━━━━━━━━━━━━━
TARGET POLICY
━━━━━━━━━━━━━━━━━━━━
Filename: {filename}
Brand: {brand}

━━━━━━━━━━━━━━━━━━━━
SCOPE
━━━━━━━━━━━━━━━━━━━━
Extract values ONLY for Plaque Psoriasis (PsO).
Ignore PsA, Crohn's, UC, RA, AS, and all other indications.
Use ONLY the retrieved chunks below — no outside knowledge.
Always return complete sentences; do not truncate answers.
If evidence is missing, return "NA", "No", or "Unspecified" per the field rules below.

━━━━━━━━━━━━━━━━━━━━
RETRIEVED CHUNKS
━━━━━━━━━━━━━━━━━━━━
{retrieved_chunks}

━━━━━━━━━━━━━━━━━━━━
PARAMETER → CHUNK ROUTING
━━━━━━━━━━━━━━━━━━━━
{routing_hints}

━━━━━━━━━━━━━━━━━━━━
FIELD RULES
━━━━━━━━━━━━━━━━━━━━

[Age]
- Return ">=N" if explicit minimum age stated (e.g. ">=6")
- Return "NA" if no age criterion.
- Return lowest age if multiple ages present.
- Return "FDA labelled age" where it is mentioned AND the policy does not specify a numerical threshold.
- Return "NA" even if "adult"/"pediatric" mentioned without a number.

[Step Therapy Requirements Documented in Policy]
- Return VERBATIM policy text describing step therapy for PsO.
- Include all bullets, sub-conditions, BSA thresholds, prior therapies, and
  "inadequate response or intolerance" / "contraindication" clauses.
- Preserve original wording. Do NOT paraphrase, summarize, or convert to Yes/No.
- If multiple passages, concatenate with " | ".
- Return "NA" only if no step therapy criteria exist for PsO.

[Number of Steps through Brands]  (integer or "NA")
[Number of Steps through Generic] (integer or "NA")
[Step through-Phototherapy]       ("Yes" / "No" / "NA")

  COUNTING LOGIC:
  Step therapy in a PA policy is built from criteria connected by AND or OR.
  AND → every connected requirement counts as a separate step.
  OR  → only ONE alternative must be satisfied. Choose the LEAST RESTRICTIVE branch.
  Sibling bullets at the same indent without explicit connectors default to OR.

  Brands step  = biologics, targeted synthetic agents, preferred branded products.
  Generic step = conventional systemics: methotrexate, cyclosporine, acitretin, topicals.
  Phototherapy (UVB, PUVA) is NEVER counted under brands or generics.

  "Step through-Phototherapy" = "Yes" ONLY if phototherapy is a MANDATORY requirement
  (AND path, no OR alternative). If phototherapy appears only inside an OR branch,
  return "No". Return "NA" if phototherapy is not mentioned.

[TB Test required]
- "Y" if policy mentions TB screening, IGRA, QuantiFERON, TST, PPD, latent TB,
  tuberculosis testing, or chest X-ray for TB.
- Else "N".

[Quantity Limits]
- Return VERBATIM the QL / Quantity Level Limit block including dose, vial size,
  units per day-supply, and any exception limits.
- Do not capture if explicitly stated as "dosage" or "dosing limit".
- "NA" if no QL stated.

[Specialist Types]
- Return a comma-separated list of specialist types explicitly mentioned.
  (e.g. Dermatologist, Rheumatologist, Gastroenterologist)
- Return "NA" if no specialist requirement is mentioned.

[Initial Authorization Duration(in-months)]
- Return integer number of months.
- Look for "initial authorization", "approval for ... months",
  "authorization of N months may be granted".
- Return "Unspecified" if no months are specified.

[Reauthorization Duration(in-months)]
- Integer months for renewal / continuation / reauthorization.
- If period in weeks, convert and round down (e.g. 16 weeks → 3 months).
- "Unspecified" if reauth exists but no duration. "NA" if no reauth at all.

[Reauthorization Required]
- "Yes" if Reauthorization Duration is non-"Unspecified" OR Requirements are non-"NA".
- "No" if BOTH are absent.

[Reauthorization Requirements Documented in Policy]
- Return VERBATIM continuation criteria text.
- Every word must be traceable to the retrieved chunk text.
- Preserve original wording. Do NOT paraphrase.
- "NA" if no reauth requirements.

━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━
Return ONLY this JSON object. No fences, no commentary.

{{
  "Age": "",
  "Step Therapy Requirements Documented in Policy": "",
  "Number of Steps through Brands": "",
  "Number of Steps through Generic": "",
  "Step through-Phototherapy": "",
  "TB Test required": "",
  "Quantity Limits": "",
  "Specialist Types": "",
  "Initial Authorization Duration(in-months)": "",
  "Reauthorization Duration(in-months)": "",
  "Reauthorization Required": "",
  "Reauthorization Requirements Documented in Policy": ""
}}
"""


def build_prompt(
    filename: str,
    brand: str,
    chunks: List[Dict[str, Any]],
    param_to_chunk_ids: Dict[str, List[int]],
) -> str:
    context_blocks = []
    for c in chunks:
        block = (
            f"CHUNK_ID: {c.get('chunk_id', '')}\n"
            f"CHUNK_TYPE: {c.get('chunk_type', '')}\n"
            f"PAGE_ID: {c.get('page_id', '')}\n"
            f"BRAND_NAME: {c.get('brand_name', '')}\n"
            f"POLICY_PARAMS: {c.get('policy_param', '')}\n\n"
            f"CONTENT:\n{c.get('content', '')}"
        )
        context_blocks.append(block.strip())
    context = "\n\n━━━━━━━━━━━━━━━━━━━━\n\n".join(context_blocks)
    routing_hints = "\n".join(f"{p} -> {ids}" for p, ids in param_to_chunk_ids.items())
    return EXTRACTION_PROMPT_TEMPLATE.format(
        filename=filename,
        brand=brand,
        retrieved_chunks=context,
        routing_hints=routing_hints,
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSON Extraction Helper
# ──────────────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[Dict]:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
        candidate = (
            candidate.replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'")
        )
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def derive_reauthorization_required(parsed: Dict) -> str:
    duration = str(parsed.get("Reauthorization Duration(in-months)", "")).strip()
    requirements = str(parsed.get("Reauthorization Requirements Documented in Policy", "")).strip()
    duration_exists = duration not in ("", "NA", "Unspecified")
    requirements_exist = requirements not in ("", "NA")
    return "Yes" if (duration_exists or requirements_exist) else "No"


def compute_access_score(parsed: Dict) -> int:
    score = 100

    def is_blank(val):
        return str(val).strip().lower() in ("", "na", "unspecified", "none")

    # Brand steps (max −30)
    brand_steps = parsed.get("Number of Steps through Brands", "NA")
    if not is_blank(brand_steps):
        try:
            n = int(brand_steps)
            score -= 10 if n == 1 else (20 if n == 2 else 30)
        except (ValueError, TypeError):
            pass

    # Generic steps (max −15)
    generic_steps = parsed.get("Number of Steps through Generic", "NA")
    if not is_blank(generic_steps):
        try:
            n = int(generic_steps)
            score -= 5 if n == 1 else (10 if n == 2 else 15)
        except (ValueError, TypeError):
            pass

    # Phototherapy (max −10)
    if str(parsed.get("Step through-Phototherapy", "")).strip().lower() == "yes":
        score -= 10

    # TB test (max −5)
    if str(parsed.get("TB Test required", "")).strip().lower() == "yes":
        score -= 5

    # Initial auth duration (max −5)
    init_auth = str(parsed.get("Initial Authorization Duration(in-months)", "")).strip()
    if not is_blank(init_auth):
        try:
            months = int(init_auth)
            if months < 6:
                score -= 5
            elif months < 12:
                score -= 3
        except (ValueError, TypeError):
            pass

    # Reauthorization (max −7)
    reauth_req = str(parsed.get("Reauthorization Required", "")).strip().lower()
    reauth_dur = str(parsed.get("Reauthorization Duration(in-months)", "")).strip()
    if reauth_req == "yes":
        if is_blank(reauth_dur):
            score -= 3
        else:
            try:
                months = int(reauth_dur)
                score -= 7 if months <= 6 else (4 if months < 12 else 0)
            except (ValueError, TypeError):
                score -= 3

    # Quantity limits (max −5)
    if not is_blank(parsed.get("Quantity Limits", "")):
        score -= 5

    # Specialist required (max −3)
    if not is_blank(parsed.get("Specialist Types", "")):
        score -= 3

    # Age (max −10 / +5 bonus)
    age = str(parsed.get("Age", "")).strip().lower()
    if not is_blank(age) and "fda" not in age:
        age_match = re.search(r"(\d+)", age)
        if age_match:
            age_num = int(age_match.group(1))
            if age_num > 18:
                score -= 10
            elif age_num <= 12:
                score += 5

    # Step therapy catch-all (−5 if step therapy present but counts unknown)
    step_therapy = str(
        parsed.get("Step Therapy Requirements Documented in Policy", "")
    ).strip().lower()
    if not is_blank(step_therapy) and step_therapy != "no":
        if is_blank(parsed.get("Number of Steps through Brands", "NA")) and \
           is_blank(parsed.get("Number of Steps through Generic", "NA")):
            score -= 5

    return max(0, min(100, round(score)))


# ──────────────────────────────────────────────────────────────────────────────
# LLM Client
# ──────────────────────────────────────────────────────────────────────────────

class GroqExtractor:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def extract(self, prompt: str, retries: int = 2, sleep_on_fail: int = 60) -> tuple[Optional[Dict], str, str]:
        """
        Returns (parsed_dict, raw_text, status)
        """
        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.choices[0].message.content.strip()
                parsed = extract_json(text)
                if parsed is None:
                    return None, text, "json_parse_failed"
                return parsed, text, "success"
            except Exception as e:
                if attempt < retries:
                    time.sleep(sleep_on_fail)
                else:
                    return None, "", f"error: {e}"
        return None, "", "error: max_retries"
