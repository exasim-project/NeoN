# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
Tests for NeoN container free functions Python bindings.

This module tests the fill and equal functions for various vector types.
"""

import neon


def test_container_imports():
    """Test that all container-related functions can be accessed."""
    assert hasattr(neon, 'fill')
    assert hasattr(neon, 'fill_range')
    assert hasattr(neon, 'equal')

    print("✓ All container free function imports successful")


# ============================================================================
# fill tests for ScalarVector
# ============================================================================

def test_fill_scalar_vector():
    """Test fill function with ScalarVector."""
    exec = neon.SerialExecutor()
    v = neon.ScalarVector(exec, 10)

    # Fill with a value
    neon.fill(v, 3.14)

    # Verify all elements are filled
    assert neon.equal(v, 3.14)

    print("✓ fill(ScalarVector, scalar) test passed")


def test_fill_range_scalar_vector():
    """Test fill_range function with ScalarVector."""
    exec = neon.SerialExecutor()
    v = neon.ScalarVector(exec, 10, 0.0)  # Initialize with zeros

    # Fill a range with a value
    neon.fill_range(v, 5.0, 2, 7)

    # The elements at indices 2-6 should be 5.0
    # We can't easily check individual elements without more bindings,
    # but we can verify the whole vector is NOT all 5.0 (since we only filled part)
    assert not neon.equal(v, 5.0)  # Not all elements are 5.0
    assert not neon.equal(v, 0.0)  # Not all elements are 0.0 anymore

    print("✓ fill_range(ScalarVector, scalar, start, end) test passed")


def test_fill_scalar_vector_zero():
    """Test fill with zero value."""
    exec = neon.SerialExecutor()
    v = neon.ScalarVector(exec, 5, 1.0)  # Initialize with ones

    neon.fill(v, 0.0)

    assert neon.equal(v, 0.0)

    print("✓ fill(ScalarVector, 0.0) test passed")


# ============================================================================
# fill tests for VectorVector (Vec3)
# ============================================================================

def test_fill_vector_vector():
    """Test fill function with VectorVector (Vec3)."""
    exec = neon.SerialExecutor()
    v = neon.VectorVector(exec, 10)

    # Fill with a Vec3 value
    fill_value = neon.Vec3(1.0, 2.0, 3.0)
    neon.fill(v, fill_value)

    # Verify all elements are filled
    assert neon.equal(v, fill_value)

    print("✓ fill(VectorVector, Vec3) test passed")


def test_fill_range_vector_vector():
    """Test fill_range function with VectorVector."""
    exec = neon.SerialExecutor()
    zero_vec = neon.Vec3(0.0, 0.0, 0.0)
    v = neon.VectorVector(exec, 10, zero_vec)

    # Fill a range with a value
    fill_value = neon.Vec3(1.0, 1.0, 1.0)
    neon.fill_range(v, fill_value, 3, 8)

    # Not all elements should be the fill value
    assert not neon.equal(v, fill_value)
    assert not neon.equal(v, zero_vec)

    print("✓ fill_range(VectorVector, Vec3, start, end) test passed")


# ============================================================================
# fill tests for LabelVector
# ============================================================================

def test_fill_label_vector():
    """Test fill function with LabelVector."""
    exec = neon.SerialExecutor()
    v = neon.LabelVector(exec, 10)

    # Fill with a label value
    neon.fill(v, 42)

    # Verify all elements are filled
    assert neon.equal(v, 42)

    print("✓ fill(LabelVector, label) test passed")


def test_fill_range_label_vector():
    """Test fill_range function with LabelVector."""
    exec = neon.SerialExecutor()
    v = neon.LabelVector(exec, 10, 0)  # Initialize with zeros

    # Fill a range with a value
    neon.fill_range(v, 99, 1, 5)

    # Not all elements should be 99 or 0
    assert not neon.equal(v, 99)
    assert not neon.equal(v, 0)

    print("✓ fill_range(LabelVector, label, start, end) test passed")


# ============================================================================
# equal tests for ScalarVector
# ============================================================================

def test_equal_scalar_vector_value():
    """Test equal function comparing ScalarVector to a value."""
    exec = neon.SerialExecutor()

    # All same value
    v1 = neon.ScalarVector(exec, 5, 7.5)
    assert neon.equal(v1, 7.5)
    assert not neon.equal(v1, 0.0)

    print("✓ equal(ScalarVector, scalar) test passed")


def test_equal_scalar_vectors():
    """Test equal function comparing two ScalarVectors."""
    exec = neon.SerialExecutor()

    v1 = neon.ScalarVector(exec, 5, 3.14)
    v2 = neon.ScalarVector(exec, 5, 3.14)
    v3 = neon.ScalarVector(exec, 5, 2.71)

    assert neon.equal(v1, v2)
    assert not neon.equal(v1, v3)

    print("✓ equal(ScalarVector, ScalarVector) test passed")


def test_equal_scalar_vectors_different_sizes():
    """Test equal function with different sized vectors."""
    exec = neon.SerialExecutor()

    v1 = neon.ScalarVector(exec, 5, 1.0)
    v2 = neon.ScalarVector(exec, 10, 1.0)

    # Different sizes should not be equal
    assert not neon.equal(v1, v2)

    print("✓ equal(ScalarVector, ScalarVector) different sizes test passed")


# ============================================================================
# equal tests for VectorVector
# ============================================================================

def test_equal_vector_vector_value():
    """Test equal function comparing VectorVector to a Vec3 value."""
    exec = neon.SerialExecutor()

    fill_val = neon.Vec3(1.0, 2.0, 3.0)
    v = neon.VectorVector(exec, 5, fill_val)

    assert neon.equal(v, fill_val)
    assert not neon.equal(v, neon.Vec3(0.0, 0.0, 0.0))

    print("✓ equal(VectorVector, Vec3) test passed")


def test_equal_vector_vectors():
    """Test equal function comparing two VectorVectors."""
    exec = neon.SerialExecutor()

    val1 = neon.Vec3(1.0, 2.0, 3.0)
    val2 = neon.Vec3(4.0, 5.0, 6.0)

    v1 = neon.VectorVector(exec, 5, val1)
    v2 = neon.VectorVector(exec, 5, val1)
    v3 = neon.VectorVector(exec, 5, val2)

    assert neon.equal(v1, v2)
    assert not neon.equal(v1, v3)

    print("✓ equal(VectorVector, VectorVector) test passed")


# ============================================================================
# equal tests for LabelVector
# ============================================================================

def test_equal_label_vector_value():
    """Test equal function comparing LabelVector to a value."""
    exec = neon.SerialExecutor()

    v = neon.LabelVector(exec, 5, 123)

    assert neon.equal(v, 123)
    assert not neon.equal(v, 0)

    print("✓ equal(LabelVector, label) test passed")


def test_equal_label_vectors():
    """Test equal function comparing two LabelVectors."""
    exec = neon.SerialExecutor()

    v1 = neon.LabelVector(exec, 5, 42)
    v2 = neon.LabelVector(exec, 5, 42)
    v3 = neon.LabelVector(exec, 5, 99)

    assert neon.equal(v1, v2)
    assert not neon.equal(v1, v3)

    print("✓ equal(LabelVector, LabelVector) test passed")


# ============================================================================
# Edge cases
# ============================================================================

def test_fill_empty_vector():
    """Test fill with an empty vector."""
    exec = neon.SerialExecutor()
    v = neon.ScalarVector(exec, 0)

    # Should not crash
    neon.fill(v, 1.0)

    assert v.empty()
    assert v.size() == 0

    print("✓ fill empty vector test passed")


def test_equal_empty_vectors():
    """Test equal with empty vectors."""
    exec = neon.SerialExecutor()

    v1 = neon.ScalarVector(exec, 0)
    v2 = neon.ScalarVector(exec, 0)

    # Empty vectors should be equal
    assert neon.equal(v1, v2)

    print("✓ equal empty vectors test passed")


if __name__ == "__main__":
    # Run all tests
    test_container_imports()

    # fill tests
    test_fill_scalar_vector()
    test_fill_range_scalar_vector()
    test_fill_scalar_vector_zero()
    test_fill_vector_vector()
    test_fill_range_vector_vector()
    test_fill_label_vector()
    test_fill_range_label_vector()

    # equal tests
    test_equal_scalar_vector_value()
    test_equal_scalar_vectors()
    test_equal_scalar_vectors_different_sizes()
    test_equal_vector_vector_value()
    test_equal_vector_vectors()
    test_equal_label_vector_value()
    test_equal_label_vectors()

    # Edge cases
    test_fill_empty_vector()
    test_equal_empty_vectors()

    print("\n" + "=" * 60)
    print("✓ All container free function binding tests passed successfully!")
    print("=" * 60)
