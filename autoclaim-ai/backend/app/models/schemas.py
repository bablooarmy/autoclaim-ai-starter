from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    TOTAL_LOSS = "TOTAL_LOSS"


class PolicyTier(str, Enum):
    COMPREHENSIVE = "COMPREHENSIVE"
    COLLISION = "COLLISION"
    LIABILITY_ONLY = "LIABILITY_ONLY"


class DamagedPart(BaseModel):
    part_name: str
    damage_type: str
    local_severity: SeverityLevel
    repair_action: str


class VisionAssessment(BaseModel):
    vehicle_type: str = "Sedan"
    damaged_parts: list[DamagedPart] = Field(default_factory=list)
    structural_risk: bool = False
    airbag_or_glass_compromised: bool = False
    visual_summary: str = ""
    visual_confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class PolicyCitation(BaseModel):
    doc_id: str
    title: str
    section: str
    excerpt: str
    score: float


class GovernanceReport(BaseModel):
    pii_redacted: bool = False
    redacted_fields: list[str] = Field(default_factory=list)
    rag_faithfulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    human_review_required: bool = False
    escalation_reasons: list[str] = Field(default_factory=list)


class AgentStepTrace(BaseModel):
    agent_name: str
    status: str
    duration_ms: int
    summary: str


class ClaimAssessmentResponse(BaseModel):
    claim_id: str
    created_at: datetime
    policy_tier: PolicyTier
    sanitized_description: str
    severity: SeverityLevel
    estimated_cost_min_usd: float
    estimated_cost_max_usd: float
    deductible_usd: float
    coverage_eligible: bool
    recommended_action: str
    vision_assessment: VisionAssessment
    policy_citations: list[PolicyCitation]
    governance: GovernanceReport
    agent_trace: list[AgentStepTrace]


class IngestRequest(BaseModel):
    title: str
    section: str
    content: str


class EvalCaseResult(BaseModel):
    case_id: str
    expected_severity: SeverityLevel
    predicted_severity: SeverityLevel
    passed: bool
    faithfulness_score: float


class EvalSuiteResponse(BaseModel):
    total_cases: int
    passed_cases: int
    accuracy: float
    avg_faithfulness: float
    results: list[EvalCaseResult]
