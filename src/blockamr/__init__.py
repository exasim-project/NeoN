# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import jax

jax.config.update("jax_enable_x64", True)

from ._blockamr import *
from .field import CellField, FaceField, Field, NodalField, PatchData
from . import dsl
from . import schemes

_default_executor = "cpu"


def set_executor(executor):
    global _default_executor
    _default_executor = executor
    jax.config.update("jax_platform_name", "gpu" if executor == "gpu" else "cpu")


def get_executor():
    return _default_executor
