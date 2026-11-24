# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from ..cpp_kernels import CppSourceAcc
from ..dsl.eqterm import EqTerm


class Source(EqTerm):
    """Pointwise source ``S(x,y,z,t) * phi``, with ``coeff_func(x, y, z, t)``
    evaluated at CELL CENTRES.
    """

    kind = "spatial"
    scheme_key = "source"

    def __init__(self, coeff_func, field, coeff=1.0):
        super().__init__(field, coeff=coeff, coefficient=coeff_func)
        self.coeff_func = coeff_func

    def build_cpp_kernel(self):
        return CppSourceAcc()
