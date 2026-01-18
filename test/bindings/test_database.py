# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for NeoN database bindings (Document, Collection, Database)."""

import neon


def test_document_creation():
    """Test Document creation and basic properties."""
    doc = neon.Document()

    # Document should have a unique ID
    doc_id = doc.id()
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0

    # Test __repr__ and __str__
    repr_str = repr(doc)
    assert "Document" in repr_str
    assert doc_id in repr_str

    str_str = str(doc)
    assert "Document" in str_str


def test_document_validation():
    """Test Document validation."""
    doc = neon.Document()

    # Validate should return True for a valid document
    is_valid = doc.validate()
    assert isinstance(is_valid, bool)


def test_database_creation():
    """Test Database creation and basic properties."""
    db = neon.Database()

    # New database should be empty
    assert db.size() == 0
    assert len(db) == 0

    # Test __repr__ and __str__
    repr_str = repr(db)
    assert "Database" in repr_str

    str_str = str(db)
    assert "Database" in str_str


def test_database_contains():
    """Test Database contains method."""
    db = neon.Database()

    # Database should not contain a non-existent collection
    assert not db.contains("nonexistent")
    assert "nonexistent" not in db


def test_document_unique_ids():
    """Test that each Document gets a unique ID."""
    doc1 = neon.Document()
    doc2 = neon.Document()

    assert doc1.id() != doc2.id()


if __name__ == "__main__":
    test_document_creation()
    test_document_validation()
    test_database_creation()
    test_database_contains()
    test_document_unique_ids()
    print("All database tests passed!")
