"""Initial schema - jobs, job_states, job_results tables

Revision ID: 001
Revises:
Create Date: 2026-02-02 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial schema."""

    # Create enum type for JobState
    job_state_enum = postgresql.ENUM(
        "pending", "processing", "completed", "failed", "cancelled", name="jobstate"
    )
    job_state_enum.create(op.get_bind(), checkfirst=True)

    # Create jobs table
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("preset", sa.String(50), nullable=False),
        sa.Column("state", job_state_enum, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column(
            "job_metadata", postgresql.JSON(astext_type=sa.Text()), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_jobs_user_id"), "jobs", ["user_id"], unique=False)

    # Create job_states table (state history)
    op.create_table(
        "job_states",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("previous_state", job_state_enum, nullable=True),
        sa.Column("new_state", job_state_enum, nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "state_metadata", postgresql.JSON(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_job_states_job_id"), "job_states", ["job_id"], unique=False
    )

    # Create job_results table
    op.create_table(
        "job_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("output", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index(
        op.f("ix_job_results_job_id"), "job_results", ["job_id"], unique=True
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_index(op.f("ix_job_results_job_id"), table_name="job_results")
    op.drop_table("job_results")
    op.drop_index(op.f("ix_job_states_job_id"), table_name="job_states")
    op.drop_table("job_states")
    op.drop_index(op.f("ix_jobs_user_id"), table_name="jobs")
    op.drop_table("jobs")

    # Drop enum type
    job_state_enum = postgresql.ENUM(
        "pending", "processing", "completed", "failed", "cancelled", name="jobstate"
    )
    job_state_enum.drop(op.get_bind(), checkfirst=True)
