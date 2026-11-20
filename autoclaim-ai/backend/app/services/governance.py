import re
from app.config import settings
from app.models.schemas import GovernanceReport, PolicyCitation, SeverityLevel

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4}(?!\w)"
)
POLICY_RE = re.compile(
    r"(?i)\b(?:policy\s*(?:no\.?|number|id|#)?\s*[:#-]?\s*)([A-Z0-9-]{6,20})\b"
)
PLATE_RE = re.compile(
    r"\b[A-Z]{2,3}[-\s]?\d{1,2}[-\s]?[A-Z]{1,3}[-\s]?\d{1,4}\b"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Redact common PII patterns before sending claim descriptions to LLMs."""
    redacted_fields: list[str] = []
    sanitized = text

    # Apply labeled policy number replacement first so digits aren't caught by other rules
    if POLICY_RE.search(sanitized):
        redacted_fields.append("policy_number")
        sanitized = POLICY_RE.sub(r"\1[REDACTED_POLICY]", sanitized)

    for label, pattern, token in [
        ("email", EMAIL_RE, "[REDACTED_EMAIL]"),
        ("phone", PHONE_RE, "[REDACTED_PHONE]"),
        ("ssn", SSN_RE, "[REDACTED_SSN]"),
        ("license_plate", PLATE_RE, "[REDACTED_PLATE]"),
    ]:
        if pattern.search(sanitized):
            redacted_fields.append(label)
            sanitized = pattern.sub(token, sanitized)

    return sanitized, redacted_fields


def compute_rag_faithfulness(answer_text: str, citations: list[PolicyCitation]) -> float:
    """Lightweight lexical-grounding faithfulness score (0.0-1.0) for local RAG governance."""
    if not citations or not answer_text.strip():
        return 0.0
    context_tokens = set(
        re.findall(r"[a-z0-9]{4,}", " ".join(c.excerpt.lower() for c in citations))
    )
    answer_tokens = set(re.findall(r"[a-z0-9]{4,}", answer_text.lower()))
    if not answer_tokens:
        return 0.0
    overlap = len(context_tokens & answer_tokens)
    return round(min(1.0, overlap / max(6, len(answer_tokens) * 0.45)), 2)


def evaluate_governance(
    sanitized_text: str,
    redacted_fields: list[str],
    severity: SeverityLevel,
    max_cost: float,
    confidence: float,
    citations: list[PolicyCitation],
    recommended_action: str,
) -> GovernanceReport:
    faithfulness = compute_rag_faithfulness(recommended_action, citations)
    escalation_reasons: list[str] = []

    if confidence < settings.min_confidence_for_auto_approve:
        escalation_reasons.append(
            f"Vision confidence ({confidence:.2f}) below threshold "
            f"({settings.min_confidence_for_auto_approve:.2f})"
        )
    if severity in (SeverityLevel.SEVERE, SeverityLevel.TOTAL_LOSS):
        escalation_reasons.append(f"Severity classified as {severity.value}; adjuster sign-off required")
    if max_cost > settings.max_auto_approve_estimate_usd:
        escalation_reasons.append(
            f"Upper repair estimate (${max_cost:,.0f}) exceeds auto-approval limit "
            f"(${settings.max_auto_approve_estimate_usd:,.0f})"
        )
    if faithfulness < 0.35:
        escalation_reasons.append("Low RAG policy grounding score; verify policy rules manually")

    return GovernanceReport(
        pii_redacted=bool(redacted_fields),
        redacted_fields=redacted_fields,
        rag_faithfulness_score=faithfulness,
        human_review_required=bool(escalation_reasons),
        escalation_reasons=escalation_reasons,
    )
