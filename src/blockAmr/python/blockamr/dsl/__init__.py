# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

# This package and ``operators`` form an unbroken import cycle: ``exp`` below pulls in
# ``operators/*``, which is imported while this ``__init__`` is still half-initialised.
# It only works because those modules import the *submodule* ``..dsl.eqterm``, which
# needs no attribute of the parent package. Switching any of them to
# ``from ..dsl import EqTerm`` makes ``import blockamr.dsl`` fail with a confusing
# partially-initialised-module error.
from . import exp
from . import imp
from .solve import solve, evaluate
from .eqterm import EqTerm
from .equation import Equation

__all__ = [
    "exp",
    "imp",
    "EqTerm",
    "Equation",
    "solve",
    "evaluate",
]
