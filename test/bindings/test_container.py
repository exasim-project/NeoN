# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
Tests for NeoN container free functions Python bindings.

This module tests the fill and equal functions for various vector types.
"""

import neon
import numpy as np
import pytest


def test_container_imports():
    """Test that all container-related functions can be accessed."""
    assert hasattr(neon, 'fill')
    assert hasattr(neon, 'fill_range')
    assert hasattr(neon, 'equal')


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

    np_v = np.asarray(v)
    assert np_v.shape == (10,)
    np_v[0:4] = 42.0
    assert np.allclose(np_v[0:4], 42.0)
    assert np.allclose(np_v[4:], 3.14)
    np_v2 = np.asarray(v)
    assert np.allclose(np_v2[0:4], 42.0)
    assert not neon.equal(v, 3.14)


def test_fill_range_scalar_vector():
    """Test fill_range function with ScalarVector."""
    exec = neon.SerialExecutor()
    v = neon.ScalarVector(exec, 10, 0.0)  # Initialize with zeros

    # Fill a range with a value
    assert neon.equal(v, 0.0)  # All elements should be 0.0
    neon.fill_range(v, 5.0, 2, 7)
    assert not neon.equal(v, 0.0)

    np_v = np.asarray(v)
    assert np.allclose(np_v[0:2], 0.0)  # The elements at indices 0-1 should be 0.0
    assert np.allclose(np_v[2:7], 5.0)


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
    assert neon.equal(v, fill_value)
    np_vec3 = np.asarray(v)
    assert np_vec3.shape == (10, 3)
    np_vec3[0:4, :] = np.array([42.0, 42.0, 42.0])
    assert np.allclose(np_vec3[0:4, :], [42.0, 42.0, 42.0])
    assert np.allclose(np_vec3[4:, :], [1.0, 2.0, 3.0])

    # Verify all elements changed filled
    assert not neon.equal(v, fill_value)


def test_fill_range_vector_vector():
    """Test fill_range function with VectorVector."""
    exec = neon.SerialExecutor()
    zero_vec = neon.Vec3(0.0, 0.0, 0.0)
    v = neon.VectorVector(exec, 10, zero_vec)

    assert neon.equal(v, zero_vec)
    # Fill a range with a value
    fill_value = neon.Vec3(1.0, 1.0, 1.0)
    neon.fill_range(v, fill_value, 3, 8)

    np_v = np.asarray(v)
    assert np.allclose(
        np_v[0:3, :], [0.0, 0.0, 0.0]
    )  # The elements at indices 0-2 should be zero_vec
    assert np.allclose(
        np_v[3:8, :], [1.0, 1.0, 1.0]
    )  # The elements at indices 3-7 should be fill_value
    assert np.allclose(
        np_v[8:, :], [0.0, 0.0, 0.0]
    )  # The elements at indices 8-9 should be zero_vec


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
    np_v = np.asarray(v)
    assert np_v.shape == (10,)
    np_v[0:4] = 99
    assert np.allclose(np_v[0:4], 99)
    assert np.allclose(np_v[4:], 42)
    np_v2 = np.asarray(v)
    assert np.allclose(np_v2[0:4], 99)
    assert not neon.equal(v, 42)


def test_fill_range_label_vector():
    """Test fill_range function with LabelVector."""
    exec = neon.SerialExecutor()
    v = neon.LabelVector(exec, 10, 0)  # Initialize with zeros
    assert neon.equal(v, 0)

    # Fill a range with a value
    neon.fill_range(v, 99, 1, 5)

    np_v = np.asarray(v)
    assert np.allclose(np_v[0:1], 0)  # The elements at index 0 should be 0
    assert np.allclose(np_v[1:5], 99)  # The elements at indices 1-4 should be 99
    assert np.allclose(np_v[5:], 0)  # The elements at indices 5-9 should be 0


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


def test_equal_scalar_vectors():
    """Test equal function comparing two ScalarVectors."""
    exec = neon.SerialExecutor()

    v1 = neon.ScalarVector(exec, 5, 3.14)
    v2 = neon.ScalarVector(exec, 5, 3.14)
    v3 = neon.ScalarVector(exec, 5, 2.71)

    assert neon.equal(v1, v2)
    assert not neon.equal(v1, v3)


def test_equal_scalar_vectors_different_sizes():
    """Test equal function with different sized vectors."""
    exec = neon.SerialExecutor()

    v1 = neon.ScalarVector(exec, 5, 1.0)
    v2 = neon.ScalarVector(exec, 10, 1.0)

    # Different sizes should not be equal
    assert not neon.equal(v1, v2)


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


# ============================================================================
# equal tests for LabelVector
# ============================================================================

def test_equal_label_vector_value():
    """Test equal function comparing LabelVector to a value."""
    exec = neon.SerialExecutor()

    v = neon.LabelVector(exec, 5, 123)

    assert neon.equal(v, 123)
    assert not neon.equal(v, 0)


def test_equal_label_vectors():
    """Test equal function comparing two LabelVectors."""
    exec = neon.SerialExecutor()

    v1 = neon.LabelVector(exec, 5, 42)
    v2 = neon.LabelVector(exec, 5, 42)
    v3 = neon.LabelVector(exec, 5, 99)

    assert neon.equal(v1, v2)
    assert not neon.equal(v1, v3)


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


def test_equal_empty_vectors():
    """Test equal with empty vectors."""
    exec = neon.SerialExecutor()

    v1 = neon.ScalarVector(exec, 0)
    v2 = neon.ScalarVector(exec, 0)

    # Empty vectors should be equal
    assert neon.equal(v1, v2)
