# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT


class Ddt:
    """Time derivative operator for explicit DSL."""

    def __init__(self, field, coeff=1.0):
        self.field = field
        self.coeff = coeff
        self._name = "Ddt"

    def __add__(self, other):
        from ..dsl.expression import Expression

        expr = Expression()
        expr.temporal_ops.append(self)
        return expr + other

    def __sub__(self, other):
        from ..dsl.expression import Expression

        expr = Expression()
        expr.temporal_ops.append(self)
        return expr - other

    def __rmul__(self, scalar):
        return Ddt(self.field, coeff=self.coeff * scalar)
