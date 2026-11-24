# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from ..schemes.laplacian_schemes import CentralDiffLaplacian
from ..dsl.eqterm import EqTerm


class Laplacian(EqTerm):
    """Laplacian operator: div(gamma * grad(phi)).

    ``gamma`` is the physical diffusivity: a number, or a ``callable(x, y, z, t)``.
    The explicit C++ kernel supports a constant gamma only; a callable that is not
    constant raises ``NotImplementedError``.
    """

    kind = "spatial"
    _scheme_operator = "laplacian"
    scheme_key = "laplacian"

    def __init__(self, gamma, field, coeff=1.0, scheme=None):
        super().__init__(
            field, coeff=coeff, coefficient=gamma, scheme=scheme or CentralDiffLaplacian()
        )
        self.gamma = gamma
        self._scheme_explicit = scheme is not None
