# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

# Steady 1D heat conduction with a constant heat source, the Python version of
# examples/steadyDiffusion.cpp
#
#     -d/dx (gamma dT/dx) = S,  x in [0, 1],  T(0) = 0,  T(1) = 1
#
# exact solution: T(x) = x + S/(2 gamma) x (1 - x)
# The problem is solved on refined meshes to check that the L2 error converges with second order.

import sys
import traceback

import numpy as np

import neon
from neon import exp, imp

DIFFUSIVITY = 1.0  # gamma
HEAT_SOURCE = 10.0  # S


def exact_solution(x):
    return x + HEAT_SOURCE / (2.0 * DIFFUSIVITY) * x * (1.0 - x)


def to_dictionary(entries):
    """Converts a nested Python dict to a neon.Dictionary"""
    d = neon.Dictionary()
    for key, value in entries.items():
        if isinstance(value, dict):
            d.insert_dict(key, to_dictionary(value))
        elif isinstance(value, list):
            d.insert_token_list(key, neon.TokenList(value))
        elif isinstance(value, bool):
            d.insert_bool(key, value)
        elif isinstance(value, int):
            d.insert_int(key, value)
        elif isinstance(value, float):
            d.insert_double(key, value)
        else:
            d.insert_string(key, value)
    return d


def solve_steady_diffusion(exec, n_cells):
    """Solves the problem on a mesh with n_cells and returns the L2 error"""
    # patch 0 is at x = 0, patch 1 at x = 1
    mesh = neon.create_1d_uniform_mesh(exec, n_cells)

    # the mesh frees its cell centres after the first solve, so copy them here
    x = np.asarray(mesh.cell_centers.copy_to_host())[:, 0].copy()

    # fixed temperature at both ends
    bcs = [
        neon.ScalarVolumeBoundary(mesh, to_dictionary({"type": "fixedValue", "fixedValue": 0.0}), 0),
        neon.ScalarVolumeBoundary(mesh, to_dictionary({"type": "fixedValue", "fixedValue": 1.0}), 1),
    ]

    # create field named "T" on the mesh and executor with the defined boundary conditions
    T = neon.ScalarVolumeField(exec, "T", mesh, bcs)
    # fill the field with a given value
    neon.fill(T.internal_vector(), 0.0)
    # ensure the boundary values are up to date
    T.correct_boundary_conditions()

    # create surface field named "gamma" for the diffusivity on the faces
    gamma = neon.ScalarSurfaceField(exec, "gamma", mesh)
    # fill the internal and the boundary faces with the diffusivity
    neon.fill(gamma.internal_vector(), DIFFUSIVITY)
    neon.fill(gamma.boundary_data_value(), DIFFUSIVITY)

    # create field named "S" for the heat source
    source = neon.ScalarVolumeField(exec, "S", mesh)
    # fill the field with the heat source
    neon.fill(source.internal_vector(), HEAT_SOURCE)

    # discretisation schemes
    fv_schemes = to_dictionary(
        {
            # Gauss theorem, linear interpolation to the faces, no non-orthogonal correction
            "laplacianSchemes": {"laplacian(gamma,T)": ["Gauss", "linear", "uncorrected"]},
            # no time derivative
            "timeIntegration": {"type": "steadyState"},
        }
    )
    # linear solver settings
    fv_solution = to_dictionary(
        {
            # conjugate gradient solver from Ginkgo, the matrix is symmetric
            "solver": "Ginkgo",
            "type": "solver::Cg",
            # stop after 1000 iterations or once the residual is below 1e-12 times the rhs norm
            "criteria": {"iteration": 1000, "relative_residual_norm": 1e-12},
        }
    )

    # implicit (imp) terms are assembled into the matrix, explicit (exp) terms into the rhs
    eqn = imp.laplacian(gamma, T) + exp.source(source)

    # solve(equation, unknown field, time, time step, schemes, solver settings)
    # the solution is written to T, time and time step have no effect for a steady problem
    neon.solve(eqn, T, 0.0, 1.0, fv_schemes, fv_solution)

    # L2 error
    T_values = np.asarray(T.internal_vector().copy_to_host())
    volumes = np.asarray(mesh.cell_volumes.copy_to_host())
    return np.sqrt(np.sum((T_values - exact_solution(x)) ** 2 * volumes))


def main():
    # runs on the GPU if NeoN was built with GPU support
    exec = neon.GPUExecutor() if neon.__has_gpu__ else neon.CPUExecutor()
    print(f"Running on {exec.name()}\n")

    print("  nCells     L2 error  order")
    prev_err = 0.0
    for n_cells in [10, 20, 40, 80]:
        err = solve_steady_diffusion(exec, n_cells)
        line = f"{n_cells:8d}{err:13.6g}"
        if not np.isfinite(err):
            print(line)
            print(f"Invalid L2 error {err}", file=sys.stderr)
            return 1
        if prev_err > 0.0:
            order = np.log2(prev_err / err)
            line += f"{order:7.6g}"
            if order < 1.8:
                print(line)
                print(f"Unexpected convergence order {order}", file=sys.stderr)
                return 1
        print(line)
        prev_err = err
    return 0


if __name__ == "__main__":
    neon.initialize()
    try:
        result = main()
    except Exception:
        # the traceback keeps the fields alive, so release it before finalize
        traceback.print_exc()
        result = 1
    finally:
        neon.finalize()
    sys.exit(result)
