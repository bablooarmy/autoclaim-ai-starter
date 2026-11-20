from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.agents.graph import build_claim_workflow
from app.config import settings
from app.db.database import ClaimRecord, get_db, init_db
from app.models.schemas import (
    ClaimAssessmentResponse,
    EvalSuiteResponse,
    IngestRequest,
    PolicyTier,
)
from app.services.eval_service import run_eval_suite
from app.services.rag_service import LocalRAGService

rag_service: LocalRAGService | None = None


def get_rag_service() -> LocalRAGService:
    global rag_service
    if rag_service is None:
        rag_service = LocalRAGService()
        rag_service.seed_default_policies()
    return rag_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    get_rag_service()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="100% Local Open-Source Multi-Agent Car Damage Severity & Insurance RAG API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.post(f"{settings.api_prefix}/claims/assess", response_model=ClaimAssessmentResponse)
async def assess_claim(
    image: UploadFile = File(...),
    description: str = Form(...),
    policy_tier: PolicyTier = Form(PolicyTier.COMPREHENSIVE),
    db: Session = Depends(get_db),
    rag: LocalRAGService = Depends(get_rag_service),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    workflow = build_claim_workflow(rag)
    result = workflow.invoke(
        {
            "image_bytes": image_bytes,
            "raw_description": description,
            "policy_tier": policy_tier,
        }
    )

    response = ClaimAssessmentResponse(
        claim_id=result["claim_id"],
        created_at=datetime.now(UTC),
        policy_tier=policy_tier,
        sanitized_description=result["sanitized_description"],
        severity=result["severity"],
        estimated_cost_min_usd=result["estimated_cost_min_usd"],
        estimated_cost_max_usd=result["estimated_cost_max_usd"],
        deductible_usd=result["deductible_usd"],
        coverage_eligible=result["coverage_eligible"],
        recommended_action=result["recommended_action"],
        vision_assessment=result["vision_assessment"],
        policy_citations=result["policy_citations"],
        governance=result["governance"],
        agent_trace=result["agent_trace"],
    )

    record = ClaimRecord(
        claim_id=response.claim_id,
        created_at=response.created_at,
        policy_tier=response.policy_tier.value,
        sanitized_description=response.sanitized_description,
        severity=response.severity.value,
        estimated_cost_min_usd=response.estimated_cost_min_usd,
        estimated_cost_max_usd=response.estimated_cost_max_usd,
        deductible_usd=response.deductible_usd,
        coverage_eligible=response.coverage_eligible,
        recommended_action=response.recommended_action,
        payload_json=response.model_dump(mode="json"),
    )
    db.add(record)
    db.commit()
    return response


@app.get(f"{settings.api_prefix}/claims", response_model=list[ClaimAssessmentResponse])
def list_claims(db: Session = Depends(get_db)):
    rows = db.query(ClaimRecord).order_by(ClaimRecord.created_at.desc()).limit(50).all()
    return [ClaimAssessmentResponse.model_validate(r.payload_json) for r in rows]


@app.post(f"{settings.api_prefix}/rag/ingest")
def ingest_policy(req: IngestRequest, rag: LocalRAGService = Depends(get_rag_service)):
    count = rag.ingest_documents([req.model_dump()])
    return {"ingested": count, "collection": rag.collection}


@app.post(f"{settings.api_prefix}/evals/run", response_model=EvalSuiteResponse)
def trigger_evals(rag: LocalRAGService = Depends(get_rag_service)):
    return run_eval_suite(rag)
