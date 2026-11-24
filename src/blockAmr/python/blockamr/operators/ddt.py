# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from ..dsl.eqterm import EqTerm


class Ddt(EqTerm):
    """Time derivative operator for explicit DSL."""

    kind = "temporal"
    scheme_key = "ddt"

    def __init__(self, field, coeff=1.0):
        super().__init__(field, coeff=coeff)
