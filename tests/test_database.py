"""
Tests for database connectivity and basic operations.
"""

import pytest
from core.data.database import Database


def test_database_initialization():
    """Test that database can be initialized."""
    db = Database()
    assert db is not None
    assert db.engine is not None


@pytest.mark.integration
def test_database_connection():
    """Test database connectivity (requires running database)."""
    db = Database()
    assert db.test_connection() is True


@pytest.mark.integration
def test_session_context_manager():
    """Test database session context manager."""
    db = Database()
    with db.get_session() as session:
        result = session.execute("SELECT 1").scalar()
        assert result == 1

