from fastapi.testclient import TestClient

from app.main import app, get_rag_service
from app.models.schemas import PolicyTier, SeverityLevel
from app.services.eval_service import run_eval_suite
from app.services.governance import redact_pii
from app.services.rag_service import LocalRAGService


def test_pii_redaction():
    raw = "My plate is KA-01-AB-1234, call 555-123-4567 or email driver@example.com."
    sanitized, fields = redact_pii(raw)
    assert "[REDACTED_EMAIL]" in sanitized
    assert "[REDACTED_PHONE]" in sanitized
    assert "[REDACTED_PLATE]" in sanitized
    assert set(fields) == {"email", "phone", "license_plate"}


def test_in_memory_rag_and_eval_suite():
    rag = LocalRAGService(in_memory=True)
    seeded = rag.seed_default_policies()
    assert seeded >= 5

    citations = rag.search("airbag deployed structural frame severe", limit=2)
    assert len(citations) == 2

    eval_report = run_eval_suite(rag)
    assert eval_report.total_cases == 4
    assert eval_report.accuracy == 1.0


def test_api_assess_and_list_claims():
    rag = LocalRAGService(in_memory=True)
    rag.seed_default_policies()
    app.dependency_overrides[get_rag_service] = lambda: rag

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/claims/assess",
            data={
                "description": "Deep dent on passenger door and cracked headlight.",
                "policy_tier": PolicyTier.COLLISION.value,
            },
            files={"image": ("damage.jpg", b"fake-jpeg-bytes", "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["severity"] == SeverityLevel.MODERATE.value
        assert len(body["agent_trace"]) == 4
        assert len(body["policy_citations"]) >= 1

        hist = client.get("/api/v1/claims")
        assert hist.status_code == 200
        assert len(hist.json()) >= 1
