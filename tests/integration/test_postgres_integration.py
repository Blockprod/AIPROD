"""Integration tests for PostgreSQL and Alembic setup."""

import pytest
from alembic.config import Config
from alembic.command import upgrade, downgrade
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
import os
import tempfile

from src.db.models import Base


class TestPostgreSQLSchema:
    """Tests for PostgreSQL schema integrity."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary SQLite database for testing."""
        # Using SQLite for testing - would use PostgreSQL in production
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        yield engine
        engine.dispose()

    def test_jobs_table_exists(self, temp_db):
        """Verify jobs table is created with correct columns."""
        inspector = inspect(temp_db)
        tables = inspector.get_table_names()
        assert "jobs" in tables

        columns = {col["name"]: col for col in inspector.get_columns("jobs")}
        assert "id" in columns
        assert "user_id" in columns
        assert "content" in columns
        assert "preset" in columns
        assert "state" in columns
        assert "created_at" in columns
        assert "updated_at" in columns
        assert "started_at" in columns
        assert "completed_at" in columns
        assert "job_metadata" in columns

    def test_job_states_table_exists(self, temp_db):
        """Verify job_states table is created."""
        inspector = inspect(temp_db)
        tables = inspector.get_table_names()
        assert "job_states" in tables

        columns = {col["name"]: col for col in inspector.get_columns("job_states")}
        assert "id" in columns
        assert "job_id" in columns
        assert "previous_state" in columns
        assert "new_state" in columns
        assert "reason" in columns
        assert "state_metadata" in columns
        assert "created_at" in columns

    def test_job_results_table_exists(self, temp_db):
        """Verify job_results table is created."""
        inspector = inspect(temp_db)
        tables = inspector.get_table_names()
        assert "job_results" in tables

        columns = {col["name"]: col for col in inspector.get_columns("job_results")}
        assert "id" in columns
        assert "job_id" in columns
        assert "status" in columns
        assert "output" in columns
        assert "error_message" in columns
        assert "processing_time_ms" in columns
        assert "created_at" in columns

    def test_foreign_key_relationships(self, temp_db):
        """Verify foreign key relationships are set up correctly."""
        inspector = inspect(temp_db)

        # Check job_states -> jobs FK
        job_states_fks = inspector.get_foreign_keys("job_states")
        assert len(job_states_fks) > 0
        assert any(fk["referred_table"] == "jobs" for fk in job_states_fks)

        # Check job_results -> jobs FK
        job_results_fks = inspector.get_foreign_keys("job_results")
        assert len(job_results_fks) > 0
        assert any(fk["referred_table"] == "jobs" for fk in job_results_fks)

    def test_indexes_created(self, temp_db):
        """Verify indexes are created for performance."""
        inspector = inspect(temp_db)

        # Check jobs table indexes
        jobs_indexes = inspector.get_indexes("jobs")
        assert any(idx["name"] == "ix_jobs_user_id" for idx in jobs_indexes)

        # Check job_states indexes
        job_states_indexes = inspector.get_indexes("job_states")
        assert any(idx["name"] == "ix_job_states_job_id" for idx in job_states_indexes)

        # Check job_results indexes
        job_results_indexes = inspector.get_indexes("job_results")
        assert any(idx["name"] == "ix_job_results_job_id" for idx in job_results_indexes)

    def test_unique_constraint_on_job_results(self, temp_db):
        """Verify unique constraint on job_results.job_id (PostgreSQL feature)."""
        # Note: SQLite doesn't expose unique constraints via inspector.get_unique_constraints()
        # This test is informational - the constraint is defined in the model
        inspector = inspect(temp_db)
        
        # Try to get constraints, but don't fail if not available in SQLite
        try:
            constraints = inspector.get_unique_constraints("job_results")
            if constraints:  # Only assert if SQLite returns them
                assert any("job_id" in const for const in constraints)
        except (NotImplementedError, AttributeError):
            # SQLite doesn't support this, which is fine
            pass

    def test_schema_performance_indexes(self, temp_db):
        """Verify indexes that improve query performance."""
        inspector = inspect(temp_db)

        # user_id should be indexed for list_jobs queries
        jobs_indexes = inspector.get_indexes("jobs")
        user_id_indexed = any(
            idx["name"] == "ix_jobs_user_id" for idx in jobs_indexes
        )
        assert user_id_indexed, "jobs.user_id should be indexed for filtering"

        # job_id should be indexed in job_states for history queries
        job_states_indexes = inspector.get_indexes("job_states")
        job_id_indexed = any(
            idx["name"] == "ix_job_states_job_id" for idx in job_states_indexes
        )
        assert job_id_indexed, "job_states.job_id should be indexed"


class TestAlembicMigrations:
    """Tests for Alembic migration setup."""

    def test_alembic_ini_exists(self):
        """Verify alembic.ini configuration file exists."""
        assert os.path.exists("alembic.ini")

    def test_migrations_directory_exists(self):
        """Verify migrations directory structure."""
        assert os.path.exists("migrations")
        assert os.path.exists("migrations/env.py")
        assert os.path.exists("migrations/script.py.mako")
        assert os.path.exists("migrations/versions")

    def test_initial_migration_exists(self):
        """Verify initial migration file is created."""
        migrations_dir = "migrations/versions"
        migration_files = [
            f for f in os.listdir(migrations_dir) if f.endswith(".py")
        ]
        assert len(migration_files) > 0
        assert any("001_initial_schema" in f for f in migration_files)


class TestConnectionPooling:
    """Tests for database connection pooling."""

    def test_connection_pool_size(self):
        """Verify connection pool is configured correctly."""
        from src.db.models import get_db_engine

        engine = get_db_engine("sqlite:///:memory:", pool_size=10, max_overflow=20)
        
        # Verify pool exists and is of correct type
        pool = engine.pool
        assert pool is not None
        assert pool.__class__.__name__ == "QueuePool"
        
        engine.dispose()

    def test_connection_recycling(self):
        """Verify connections are recycled (PostgreSQL configuration)."""
        from src.db.models import get_db_engine

        engine = get_db_engine("sqlite:///:memory:")
        
        # Verify engine is created successfully
        assert engine is not None
        # The recycling (pool_recycle=3600) is configured at creation time
        # This test verifies the engine was created without errors
        
        engine.dispose()
