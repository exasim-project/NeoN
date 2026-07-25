// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include <AMReX.H>

#include "bench/kokkos_bench.hpp"

namespace nb = nanobind;

void registerInit(nb::module_& m)
{
    m.def(
        "initialize",
        []()
        {
            int argc = 0;
            char** argv = nullptr;
            amrex::Initialize(argc, argv);
            // Kokkos after AMReX (AMReX creates the CUDA context) and, in
            // finalize(), before amrex::Finalize() tears that context down -- a
            // Kokkos teardown afterwards hits CUDA error 709.
            //
            // Kokkos cannot be re-initialized once finalized, so a second
            // sequential runtime() block in one process leaves it unavailable;
            // the Kokkos entry points raise rather than crash in that case.
            blockamr::bench::kokkosInitialize();
        }
    );
    m.def(
        "finalize",
        []()
        {
            blockamr::bench::kokkosFinalize();
            amrex::Finalize();
        }
    );
}
