from app.agents.graph import build_claim_workflow
from app.models.schemas import EvalCaseResult, EvalSuiteResponse, PolicyTier, SeverityLevel
from app.services.rag_service import LocalRAGService

GOLDEN_CASES = [
    {
        "case_id": "EVAL-001",
        "description": "Light scratch on rear bumper cover from parking lot scuff.",
        "policy_tier": PolicyTier.COMPREHENSIVE,
        "expected_severity": SeverityLevel.MINOR,
    },
    {
        "case_id": "EVAL-002",
        "description": "Deep dent on passenger door and cracked headlight after low-speed impact.",
        "policy_tier": PolicyTier.COLLISION,
        "expected_severity": SeverityLevel.MODERATE,
    },
    {
        "case_id": "EVAL-003",
        "description": "Front frame bent, radiator crushed, and driver airbag deployed.",
        "policy_tier": PolicyTier.COLLISION,
        "expected_severity": SeverityLevel.SEVERE,
    },
    {
        "case_id": "EVAL-004",
        "description": "Vehicle submerged in floodwater with rollover roof crush.",
        "policy_tier": PolicyTier.COMPREHENSIVE,
        "expected_severity": SeverityLevel.TOTAL_LOSS,
    },
]


def run_eval_suite(rag_service: LocalRAGService) -> EvalSuiteResponse:
    graph = build_claim_workflow(rag_service)
    results: list[EvalCaseResult] = []
    dummy_png = b"\x89PNG\r\n\x1a\n"

    for case in GOLDEN_CASES:
        out = graph.invoke(
            {
                "image_bytes": dummy_png,
                "raw_description": case["description"],
                "policy_tier": case["policy_tier"],
            }
        )
        predicted = out["severity"]
        faithfulness = out["governance"].rag_faithfulness_score
        results.append(
            EvalCaseResult(
                case_id=case["case_id"],
                expected_severity=case["expected_severity"],
                predicted_severity=predicted,
                passed=predicted == case["expected_severity"],
                faithfulness_score=faithfulness,
            )
        )

    passed = sum(1 for r in results if r.passed)
    avg_faith = round(sum(r.faithfulness_score for r in results) / len(results), 2)
    return EvalSuiteResponse(
        total_cases=len(results),
        passed_cases=passed,
        accuracy=round(passed / len(results), 2),
        avg_faithfulness=avg_faith,
        results=results,
    )
