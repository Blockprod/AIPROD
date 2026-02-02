"""SQLAlchemy models for job persistence."""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Text,
    JSON,
    Enum,
    ForeignKey,
    create_engine,
    event,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import QueuePool
import enum
import json

Base = declarative_base()


def utc_now():
    """Return current UTC time - SQLAlchemy compatible."""
    return datetime.now(timezone.utc)


class JobState(str, enum.Enum):
    """Job execution states."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """Job model for persistent storage."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    preset = Column(String(50), nullable=False)
    state = Column(Enum(JobState), default=JobState.PENDING, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    job_metadata = Column(JSON, default={}, nullable=False)

    # Relationships
    state_history = relationship(
        "JobStateRecord", back_populates="job", cascade="all, delete-orphan"
    )
    results = relationship(
        "JobResult", back_populates="job", cascade="all, delete-orphan", uselist=False
    )

    def __repr__(self):
        return f"<Job id={self.id} state={self.state}>"

    def to_dict(self):
        """Convert to dictionary."""
        created_at_val = getattr(self, "created_at", None)
        updated_at_val = getattr(self, "updated_at", None)
        started_at_val = getattr(self, "started_at", None)
        completed_at_val = getattr(self, "completed_at", None)

        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "preset": self.preset,
            "state": (
                self.state.value if isinstance(self.state, JobState) else self.state
            ),
            "created_at": created_at_val.isoformat() if created_at_val else None,
            "updated_at": updated_at_val.isoformat() if updated_at_val else None,
            "started_at": started_at_val.isoformat() if started_at_val else None,
            "completed_at": completed_at_val.isoformat() if completed_at_val else None,
            "metadata": self.job_metadata,
        }


class JobStateRecord(Base):
    """Job state history for audit trail."""

    __tablename__ = "job_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    previous_state = Column(Enum(JobState), nullable=True)
    new_state = Column(Enum(JobState), nullable=False)
    reason = Column(Text, nullable=True)
    state_metadata = Column(JSON, default={}, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    job = relationship("Job", back_populates="state_history")

    def __repr__(self):
        return f"<JobStateRecord job_id={self.job_id} {self.previous_state}→{self.new_state}>"


class JobResult(Base):
    """Job results storage."""

    __tablename__ = "job_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    status = Column(String(20), nullable=False)  # success, error, timeout
    output = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    job = relationship("Job", back_populates="results")

    def __repr__(self):
        return f"<JobResult job_id={self.job_id} status={self.status}>"

    def to_dict(self):
        """Convert to dictionary."""
        created_at_val = getattr(self, "created_at", None)

        return {
            "job_id": self.job_id,
            "status": self.status,
            "output": self.output,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
            "created_at": created_at_val.isoformat() if created_at_val else None,
        }


def get_db_engine(db_url: str, pool_size: int = 10, max_overflow: int = 20):
    """Create database engine with pooling."""
    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=False,
        pool_recycle=3600,  # Recycle connections after 1 hour
    )
    return engine


def get_session_factory(db_url: str):
    """Get SQLAlchemy session factory."""
    engine = get_db_engine(db_url)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    return Session, engine


def init_db(db_url: str):
    """Initialize database and create all tables."""
    engine = get_db_engine(db_url)
    Base.metadata.create_all(engine)
    return engine
