import time
from typing import TypedDict
import uuid

from langgraph.graph import END, StateGraph

from app.models.schemas import (
    AgentStepTrace,
    GovernanceReport,
    PolicyCitation,
    PolicyTier,
    SeverityLevel,
    VisionAssessment,
)
from app.services.governance import evaluate_governance, redact_pii
from app.services.ollama_client import adjudicate_with_ollama, analyze_damage_with_ollama
from app.services.rag_service import LocalRAGService


class ClaimState(TypedDict, total=False):
    claim_id: str
    image_bytes: bytes
    raw_description: str
    sanitized_description: str
    redacted_fields: list[str]
    policy_tier: PolicyTier
    vision_assessment: VisionAssessment
    policy_citations: list[PolicyCitation]
    severity: SeverityLevel
    estimated_cost_min_usd: float
    estimated_cost_max_usd: float
    deductible_usd: float
    coverage_eligible: bool
    recommended_action: str
    governance: GovernanceReport
    agent_trace: list[AgentStepTrace]


SEVERITY_COST_BANDS: dict[SeverityLevel, tuple[float, float]] = {
    SeverityLevel.MINOR: (300.0, 1200.0),
    SeverityLevel.MODERATE: (1200.0, 4500.0),
    SeverityLevel.SEVERE: (4500.0, 14500.0),
    SeverityLevel.TOTAL_LOSS: (15000.0, 35000.0),
}

DEDUCTIBLE_BY_TIER: dict[PolicyTier, float] = {
    PolicyTier.COMPREHENSIVE: 500.0,
    PolicyTier.COLLISION: 1000.0,
    PolicyTier.LIABILITY_ONLY: 0.0,
}


def build_claim_workflow(rag_service: LocalRAGService):
    def guardrail_intake_node(state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        sanitized, redacted = redact_pii(state["raw_description"])
        trace = list(state.get("agent_trace", []))
        trace.append(
            AgentStepTrace(
                agent_name="Guardrail & Intake Agent",
                status="completed",
                duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
                summary=f"PII scan complete; redacted={redacted or 'none'}.",
            )
        )
        return {
            "claim_id": state.get("claim_id") or f"CLM-{uuid.uuid4().hex[:8].upper()}",
            "sanitized_description": sanitized,
            "redacted_fields": redacted,
            "agent_trace": trace,
        }

    def vision_assessor_node(state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        vision = analyze_damage_with_ollama(
            state["image_bytes"], state["sanitized_description"]
        )
        trace = list(state.get("agent_trace", []))
        trace.append(
            AgentStepTrace(
                agent_name="Vision Damage Assessor",
                status="completed",
                duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
                summary=f"Identified {len(vision.damaged_parts)} damaged area(s); "
                f"confidence={vision.visual_confidence:.2f}.",
            )
        )
        return {"vision_assessment": vision, "agent_trace": trace}

    def policy_rag_node(state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        vision = state["vision_assessment"]
        parts_str = ", ".join(p.part_name for p in vision.damaged_parts) or "vehicle exterior"
        query = (
            f"{state['policy_tier'].value} coverage deductible severity rubric "
            f"{parts_str} {vision.visual_summary}"
        )
        citations = rag_service.search(query, limit=3)
        trace = list(state.get("agent_trace", []))
        trace.append(
            AgentStepTrace(
                agent_name="Policy & Repair RAG Agent",
                status="completed",
                duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
                summary=f"Retrieved {len(citations)} policy section(s) from Qdrant.",
            )
        )
        return {"policy_citations": citations, "agent_trace": trace}

    def severity_adjudicator_node(state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        vision = state["vision_assessment"]
        citations = state["policy_citations"]
        tier = state["policy_tier"]

        part_severities = [p.local_severity for p in vision.damaged_parts]
        order = [
            SeverityLevel.MINOR,
            SeverityLevel.MODERATE,
            SeverityLevel.SEVERE,
            SeverityLevel.TOTAL_LOSS,
        ]
        default_severity = (
            max(part_severities, key=lambda s: order.index(s))
            if part_severities
            else SeverityLevel.MINOR
        )
        if vision.structural_risk and order.index(default_severity) < order.index(SeverityLevel.SEVERE):
            default_severity = SeverityLevel.SEVERE

        cost_min, cost_max = SEVERITY_COST_BANDS[default_severity]
        deductible = DEDUCTIBLE_BY_TIER[tier]
        eligible = tier != PolicyTier.LIABILITY_ONLY

        rag_context = "\n".join(f"[{c.doc_id} - {c.section}]: {c.excerpt}" for c in citations)
        prompt = (
            "You are the Lead Insurance Severity Adjudicator. Given the visual damage summary, "
            "policy tier, and retrieved Qdrant policy rules, return ONLY JSON with keys: "
            "severity (MINOR|MODERATE|SEVERE|TOTAL_LOSS), estimated_cost_min_usd (number), "
            "estimated_cost_max_usd (number), recommended_action (string citing policy rules).\n\n"
            f"Policy Tier: {tier.value}\n"
            f"Visual Summary: {vision.visual_summary}\n"
            f"Structural Risk: {vision.structural_risk}\n"
            f"Policy Context:\n{rag_context}\n"
        )
        llm_decision = adjudicate_with_ollama(prompt)

        severity = default_severity
        recommended_action = (
            f"Classified as {severity.value} under {tier.value} policy "
            f"(${deductible:,.0f} deductible). "
            + (
                "Escalate for licensed adjuster inspection."
                if severity in (SeverityLevel.SEVERE, SeverityLevel.TOTAL_LOSS)
                else "Eligible for repair shop estimate or fast-track settlement."
                if eligible
                else "Own-vehicle damage is not covered under Liability-Only policies."
            )
        )

        if llm_decision:
            try:
                severity = SeverityLevel(llm_decision.get("severity", default_severity.value))
                cost_min = float(llm_decision.get("estimated_cost_min_usd", cost_min))
                cost_max = float(llm_decision.get("estimated_cost_max_usd", cost_max))
                if llm_decision.get("recommended_action"):
                    recommended_action = str(llm_decision["recommended_action"])
            except Exception:
                pass

        governance = evaluate_governance(
            sanitized_text=state["sanitized_description"],
            redacted_fields=state.get("redacted_fields", []),
            severity=severity,
            max_cost=cost_max,
            confidence=vision.visual_confidence,
            citations=citations,
            recommended_action=recommended_action,
        )

        trace = list(state.get("agent_trace", []))
        trace.append(
            AgentStepTrace(
                agent_name="Severity Adjudicator & Eval Judge",
                status="completed",
                duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
                summary=f"Severity={severity.value}, Faithfulness={governance.rag_faithfulness_score:.2f}, "
                f"HumanReview={governance.human_review_required}.",
            )
        )

        return {
            "severity": severity,
            "estimated_cost_min_usd": cost_min,
            "estimated_cost_max_usd": cost_max,
            "deductible_usd": deductible,
            "coverage_eligible": eligible,
            "recommended_action": recommended_action,
            "governance": governance,
            "agent_trace": trace,
        }

    workflow = StateGraph(ClaimState)
    workflow.add_node("guardrail_intake", guardrail_intake_node)
    workflow.add_node("vision_assessor", vision_assessor_node)
    workflow.add_node("policy_rag", policy_rag_node)
    workflow.add_node("severity_adjudicator", severity_adjudicator_node)

    workflow.set_entry_point("guardrail_intake")
    workflow.add_edge("guardrail_intake", "vision_assessor")
    workflow.add_edge("vision_assessor", "policy_rag")
    workflow.add_edge("policy_rag", "severity_adjudicator")
    workflow.add_edge("severity_adjudicator", END)

    return workflow.compile()
