# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from neon import ScalarVector, fill, equal, GPUExecutor
import numpy as np
import neon
import pytest


@pytest.mark.gpu
def test_scalar_vector_gpu():
    """Test ScalarVector functionality on GPU."""
    exec = GPUExecutor()
    v = ScalarVector(exec, 10, 1.0)
    assert v.size() == 10
    assert equal(v, 1.0)

    # Fill with a value
    fill(v, 3.14)

    # Verify all elements are filled
    assert equal(v, 3.14)
    neon.fill_range(v, 5.0, 2, 7)
    host_v = v.copy_to_host()
    np_host = np.asarray(host_v)
    assert np.allclose(np_host[0:2], 3.14)
    assert np.allclose(np_host[2:7], 5.0)
    assert np.allclose(np_host[7:], 3.14)
