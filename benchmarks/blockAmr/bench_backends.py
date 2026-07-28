# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""jax vs cpp explicit-backend performance benchmark, driven from disk.

One momentum equation is built once via the Equation API::

    UEqn = Equation(exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu, U),
                    schemes=<from system/fvSchemes>)

and re-solved with only the ``solution["backend"]`` key swapped between
``"jax"`` and ``"cpp"``. The discretisation schemes (fvSchemes) and the
solution approach (fvSolution) are READ FROM DISK, so the benchmark is
authored, not hardcoded. Parameterised over cell count and max grid size
(and scheme, though scheme is normally fixed by the case file).

Run::

    python benchmarks/blockAmr/bench_backends.py \
        --n-cell 32 64 --max-size 16 32 --steps 200
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

# GPU memory split: jax and AMReX share one device, so their totals must stay
# below 100%. Give jax a fixed 35% (preallocated) and let AMReX grow on demand
# into the rest (arena init 0 → no up-front land-grab). Without this jax is
# starved and OOMs. Set before importing jax / blockamr.
JAX_MEM_FRACTION = os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

import jax.numpy as jnp
import numpy as np
import pydantic
import yaml

import blockamr
from blockamr.dsl import exp
from blockamr.dsl.equation import Equation
from blockamr.field import CellField, FaceField
from blockamr.mesh import Mesh
from blockamr.operators.div import update_face_fluxes
from blockamr.schemes.registry import resolve

CASE = Path(__file__).parent / "cases" / "taylorGreen"
NU = 0.01


# ---------------------------------------------------------------------------
# Disk loading — pydantic-validated models over trivial YAML. neon has no
# fvSchemes/fvSolution reader yet; these models ARE the schema. A bad backend
# name or malformed block fails at load time, not deep in a solve. Returns the
# plain dict shapes the Equation API already consumes.
# ---------------------------------------------------------------------------
class FvSchemes(pydantic.RootModel[dict[str, str]]):
    """fvSchemes: scheme_key -> registry scheme name, e.g. ``div(phi,U): vanLeer``."""


class SolverBlock(pydantic.BaseModel):
    """fvSolution.solvers[field]: linear-solver + backend. Extra keys allowed."""

    model_config = pydantic.ConfigDict(extra="allow")
    backend: Literal["jax", "cpp"] = "jax"


class FvSolution(pydantic.BaseModel):
    solvers: dict[str, SolverBlock]


def read_fv_schemes(case: Path) -> dict[str, str]:
    data = yaml.safe_load((case / "system" / "fvSchemes.yaml").read_text())
    return FvSchemes.model_validate(data).root


def read_fv_solution(case: Path, field: str) -> dict:
    data = yaml.safe_load((case / "system" / "fvSolution.yaml").read_text())
    return FvSolution.model_validate(data).solvers[field].model_dump()


# ---------------------------------------------------------------------------
# Case setup
# ---------------------------------------------------------------------------
def _tg_vel(x, y, z, t):
    """Divergence-free Taylor-Green face velocity (steady advecting flux)."""
    u = jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    v = -jnp.cos(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
    w = jnp.zeros_like(z)
    return u, v, w


def build_mesh(n_cell: int, max_size: int) -> Mesh:
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])  # triply periodic
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom)


def _seed_U(U: CellField, mesh: Mesh, seed: int) -> None:
    """Fill U's valid cells with reproducible random data (same across backends)."""
    rng = np.random.default_rng(seed)
    for lev in range(mesh.n_levels()):
        for mfi in blockamr.MFIterator(U.mf[lev]):
            host = U.mf[lev].copy_to_host(mfi)
            U.mf[lev].copy_from(mfi, rng.standard_normal(host.shape))
        U.fill_patch(lev, 0.0)


def _sync(U: CellField) -> None:
    """Force device completion by reading one valid box back to host."""
    for mfi in blockamr.MFIterator(U.mf[0]):
        U.mf[0].copy_to_host(mfi)
        break


def _snapshot(U: CellField, mesh: Mesh) -> list:
    return [U.mf[0].copy_to_host(mfi).copy() for mfi in blockamr.MFIterator(U.mf[0])]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
@dataclass
class Result:
    backend: str
    n_cell: int
    max_size: int
    compile_ms: float
    per_step_ms: float
    mcell_updates_per_s: float
    snapshot: list


def bench(n_cell: int, max_size: int, backends, steps: int, warmup: int, seed: int):
    schemes = read_fv_schemes(CASE)
    sol_base = read_fv_solution(CASE, "U")

    # ngrow from the widest disk-selected stencil (div scheme + laplacian).
    div_scheme = resolve("div", schemes["div(phi,U)"])()
    ngrow = max(div_scheme.stencil_width, 1)

    results = []
    for backend in backends:
        mesh = build_mesh(n_cell, max_size)
        U = CellField(mesh, ncomp=3, ngrow=ngrow, name="U")
        phi = FaceField(mesh, ncomp=1, ngrow=ngrow, name="phi")
        _seed_U(U, mesh, seed)
        for lev in range(mesh.n_levels()):
            update_face_fluxes(phi[lev], _tg_vel, mesh.geom(lev), 0.0)

        # ONE equation, built once from disk-driven schemes.
        UEqn = Equation(
            exp.ddt(U) + exp.div(phi, U) - exp.laplacian(NU, U), schemes=schemes
        )
        sol = {**sol_base, "backend": backend}
        dt = 0.25 * (1.0 / n_cell)  # CFL-ish, exactly representable

        t0 = perf_counter()
        UEqn.solve(dt=dt, t=0.0, solution=sol)  # first solve = compile / kernel build
        _sync(U)
        compile_ms = 1e3 * (perf_counter() - t0)

        for _ in range(warmup):
            UEqn.solve(dt=dt, t=0.0, solution=sol)
        _sync(U)

        samples = []
        for _ in range(steps):
            t0 = perf_counter()
            UEqn.solve(dt=dt, t=0.0, solution=sol)
            _sync(U)
            samples.append(perf_counter() - t0)

        per_step = statistics.median(samples)
        n_cells_total = n_cell ** 3
        results.append(
            Result(
                backend=backend,
                n_cell=n_cell,
                max_size=max_size,
                compile_ms=compile_ms,
                per_step_ms=1e3 * per_step,
                mcell_updates_per_s=n_cells_total / per_step / 1e6,
                snapshot=_snapshot(U, mesh),
            )
        )
    return results


def _parity(results, rtol=1e-6, atol=1e-9):
    # Loose multi-step tolerance: the jax fused kernel casts dt/coeff to float32
    # (see test_backend_parity docstring); over many steps this accumulates to
    # ~1e-8. A single-step run agrees to ~1e-12.
    if len(results) < 2:
        return "n/a (single backend)"
    ref = results[0]
    for r in results[1:]:
        for a, b in zip(ref.snapshot, r.snapshot):
            if not np.allclose(a, b, rtol=rtol, atol=atol):
                md = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                return f"FAIL {ref.backend} vs {r.backend} (max|Δ|={md:.2e})"
    return "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cell", type=int, nargs="+", default=[32])
    ap.add_argument("--max-size", type=int, nargs="+", default=[16])
    ap.add_argument("--backends", nargs="+", default=["jax", "cpp"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()

    print(
        f"{'n_cell':>7} {'max_size':>9} {'backend':>8} {'compile_ms':>11} "
        f"{'per_step_ms':>12} {'Mcell_upd/s':>12}  parity"
    )
    with blockamr.runtime():
        for n_cell in args.n_cell:
            for max_size in args.max_size:
                res = bench(
                    n_cell, max_size, args.backends, args.steps, args.warmup, args.seed
                )
                verdict = _parity(res)
                for r in res:
                    print(
                        f"{r.n_cell:>7} {r.max_size:>9} {r.backend:>8} "
                        f"{r.compile_ms:>11.1f} {r.per_step_ms:>12.3f} "
                        f"{r.mcell_updates_per_s:>12.1f}  {verdict}"
                    )


if __name__ == "__main__":
    main()
