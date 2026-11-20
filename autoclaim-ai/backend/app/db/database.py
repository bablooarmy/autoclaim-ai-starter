from datetime import UTC, datetime
from sqlalchemy import JSON, Boolean, Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


class ClaimRecord(Base):
    __tablename__ = "claims"

    claim_id = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    policy_tier = Column(String(32), nullable=False)
    sanitized_description = Column(Text, nullable=False)
    severity = Column(String(32), nullable=False)
    estimated_cost_min_usd = Column(Float, nullable=False)
    estimated_cost_max_usd = Column(Float, nullable=False)
    deductible_usd = Column(Float, nullable=False)
    coverage_eligible = Column(Boolean, nullable=False)
    recommended_action = Column(Text, nullable=False)
    payload_json = Column(JSON, nullable=False)


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
