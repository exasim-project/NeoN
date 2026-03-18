import jax

jax.config.update("jax_enable_x64", True)

from ._blockamr import *
from .field import CellField, FaceField, Field, NodalField, PatchData
from . import dsl
from . import schemes
