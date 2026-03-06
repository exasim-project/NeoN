# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""
Tests for NeoN scalar Python bindings.

This module tests the scalar type constants and functions.
"""

import neon


def test_scalar_imports():
    """Test that all scalar-related attributes can be accessed."""
    # Constants
    assert hasattr(neon, 'ROOTVSMALL')
    assert hasattr(neon, 'DOUBLE_PRECISION')

    # Functions
    assert hasattr(neon, 'scalar_mag')
    assert hasattr(neon, 'scalar_one')
    assert hasattr(neon, 'scalar_zero')


def test_rootvsmall():
    """Test ROOTVSMALL constant."""
    assert neon.ROOTVSMALL > 0
    assert neon.ROOTVSMALL < 1e-10  # Should be very small


def test_double_precision_flag():
    """Test DOUBLE_PRECISION flag."""
    # Should be a boolean
    assert isinstance(neon.DOUBLE_PRECISION, bool)


def test_scalar_one():
    """Test scalar_one function."""
    one = neon.scalar_one()
    assert one == 1.0


def test_scalar_zero():
    """Test scalar_zero function."""
    zero = neon.scalar_zero()
    assert zero == 0.0


def test_scalar_mag_positive():
    """Test scalar_mag with positive values."""
    assert neon.scalar_mag(5.0) == 5.0
    assert neon.scalar_mag(3.14159) == 3.14159
    assert neon.scalar_mag(0.001) == 0.001


def test_scalar_mag_negative():
    """Test scalar_mag with negative values."""
    assert neon.scalar_mag(-5.0) == 5.0
    assert neon.scalar_mag(-3.14159) == 3.14159
    assert neon.scalar_mag(-0.001) == 0.001


def test_scalar_mag_zero():
    """Test scalar_mag with zero."""
    assert neon.scalar_mag(0.0) == 0.0


def test_scalar_mag_special_values():
    """Test scalar_mag with special values."""
    import math

    # Very small values
    small = 1e-15
    assert abs(neon.scalar_mag(small) - small) < 1e-20
    assert abs(neon.scalar_mag(-small) - small) < 1e-20

    # Large values
    large = 1e15
    assert abs(neon.scalar_mag(large) - large) < 1e5
    assert abs(neon.scalar_mag(-large) - large) < 1e5


if __name__ == "__main__":
    # Run all tests
    test_scalar_imports()
    test_rootvsmall()
    test_double_precision_flag()
    test_scalar_one()
    test_scalar_zero()
    test_scalar_mag_positive()
    test_scalar_mag_negative()
    test_scalar_mag_zero()
    test_scalar_mag_special_values()
