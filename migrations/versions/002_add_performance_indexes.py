"""Add performance indexes to database tables

Revision ID: 002_add_performance_indexes
Revises: 001_initial_schema
Create Date: 2026-02-04 17:45:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002_add_performance_indexes'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def upgrade():
    """Add performance indexes"""
    # Job table indexes
    op.create_index(
        'idx_jobs_status',
        'jobs',
        ['status'],
        unique=False,
        postgresql_concurrently=True
    )
    
    op.create_index(
        'idx_jobs_created_at',
        'jobs',
        ['created_at'],
        unique=False,
        postgresql_concurrently=True
    )
    
    op.create_index(
        'idx_jobs_user_id',
        'jobs',
        ['user_id'],
        unique=False,
        postgresql_concurrently=True
    )
    
    op.create_index(
        'idx_jobs_user_status',
        'jobs',
        ['user_id', 'status'],
        unique=False,
        postgresql_concurrently=True
    )
    
    # Results table indexes
    op.create_index(
        'idx_results_job_id',
        'results',
        ['job_id'],
        unique=False,
        postgresql_concurrently=True
    )
    
    op.create_index(
        'idx_results_created_at',
        'results',
        ['created_at'],
        unique=False,
        postgresql_concurrently=True
    )
    
    # Pipeline jobs indexes
    op.create_index(
        'idx_pipeline_jobs_status',
        'pipeline_jobs',
        ['status'],
        unique=False,
        postgresql_concurrently=True
    )
    
    op.create_index(
        'idx_pipeline_jobs_created_at',
        'pipeline_jobs',
        ['created_at'],
        unique=False,
        postgresql_concurrently=True
    )


def downgrade():
    """Remove performance indexes"""
    # Remove in reverse order
    op.drop_index('idx_pipeline_jobs_created_at', table_name='pipeline_jobs', postgresql_concurrently=True)
    op.drop_index('idx_pipeline_jobs_status', table_name='pipeline_jobs', postgresql_concurrently=True)
    op.drop_index('idx_results_created_at', table_name='results', postgresql_concurrently=True)
    op.drop_index('idx_results_job_id', table_name='results', postgresql_concurrently=True)
    op.drop_index('idx_jobs_user_status', table_name='jobs', postgresql_concurrently=True)
    op.drop_index('idx_jobs_user_id', table_name='jobs', postgresql_concurrently=True)
    op.drop_index('idx_jobs_created_at', table_name='jobs', postgresql_concurrently=True)
    op.drop_index('idx_jobs_status', table_name='jobs', postgresql_concurrently=True)
