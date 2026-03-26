# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

def _is_temporal_op(obj):
    return hasattr(obj, "_name") and obj._name == "Ddt"


def _is_spatial_op(obj):
    return hasattr(obj, "build_kernel")


class Expression:
    """Composable expression for explicit PDE operators."""

    def __init__(self):
        self.temporal_ops = []
        self.spatial_ops = []

    @property
    def required_ngrow(self):
        """Minimum ghost-cell count required by the widest stencil."""
        return max(
            (getattr(getattr(op, 'scheme', None), 'stencil_width', 1)
             for op in self.spatial_ops),
            default=1,
        )

    def __add__(self, other):
        if isinstance(other, Expression):
            result = Expression()
            result.temporal_ops = self.temporal_ops + other.temporal_ops
            result.spatial_ops = self.spatial_ops + other.spatial_ops
            return result
        if _is_temporal_op(other):
            self.temporal_ops.append(other)
            return self
        if _is_spatial_op(other):
            self.spatial_ops.append(other)
            return self
        return NotImplemented

    def __sub__(self, other):
        if _is_temporal_op(other) or _is_spatial_op(other):
            negated = -1.0 * other  # uses __rmul__, creates a copy
            return self.__add__(negated)
        return NotImplemented
