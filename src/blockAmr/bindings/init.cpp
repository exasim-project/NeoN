// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <AMReX.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>

#include "NeoN/blockAmr/core/runtime.hpp"

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
            blockamr::kokkosInitialize();
        }
    );
    m.def(
        "finalize",
        []()
        {
            blockamr::kokkosFinalize();
            amrex::Finalize();
        }
    );
    m.def(
        "n_ranks",
        []() { return amrex::ParallelContext::NProcsSub(); },
        "Number of MPI ranks in the current AMReX communicator (1 without MPI).\n\n"
        "Only valid inside a runtime() block -- AMReX has no communicator before\n"
        "Initialize(), which is why a skipif marker (evaluated at collection time)\n"
        "still has to read the launcher's environment instead."
    );
    m.def(
        "allreduce_sum",
        [](nb::ndarray<double, nb::c_contig, nb::device::cpu> a)
        {
            amrex::ParallelAllReduce::Sum(
                a.data(), static_cast<int>(a.size()), amrex::ParallelContext::CommunicatorSub()
            );
        },
        nb::arg("array"),
        "Sum a contiguous float64 host array across ranks, IN PLACE.\n\n"
        "AMReX's own communicator, so a reduction taken here is on the same one the\n"
        "solvers use (ParallelContext::CommunicatorSub) -- which is the point: the\n"
        "alternative, mpi4py's COMM_WORLD, is a second MPI binding in the process\n"
        "whose communicator only coincides with AMReX's by default and which is a\n"
        "dependency the package does not otherwise have.\n\n"
        "Collective: every rank must call it, with the same size. A no-op on one rank."
    );
}
