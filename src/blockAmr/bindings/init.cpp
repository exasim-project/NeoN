// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <AMReX.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelReduce.H>

#include "NeoN/blockAmr/core/runtime.hpp"

namespace nb = nanobind;

namespace
{

// Kokkos must finalize before AMReX (owner of the CUDA context), else teardown hits CUDA error 709.
void finalizeRuntime()
{
    if (amrex::Initialized())
    {
        blockamr::kokkosFinalize();
        amrex::Finalize();
    }
}

} // namespace

void registerInit(nb::module_& m)
{
    m.def(
        "initialize",
        []()
        {
            int argc = 0;
            char** argv = nullptr;
            amrex::Initialize(argc, argv);
            // Kokkos must init after AMReX (owner of the CUDA context). Kokkos cannot be
            // re-initialized once finalized, so a second runtime() block in one process leaves it
            // unavailable (entries raise).
            blockamr::kokkosInitialize();
            // A runtime left open (e.g. one never exited) is finalized at process exit. A C atexit
            // handler runs after Python's teardown has freed what it frees, and -- registered after
            // the CUDA runtime and Kokkos statics came up -- before their teardown.
            static const bool registered = std::atexit(finalizeRuntime) == 0;
            (void)registered;
        }
    );
    m.def("finalize", &finalizeRuntime);
    m.def(
        "n_ranks",
        []() { return amrex::ParallelContext::NProcsSub(); },
        "Number of MPI ranks in the current AMReX communicator (1 without MPI).\n\n"
        "Only valid inside a runtime() block -- AMReX has no communicator before Initialize()."
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
        "Reduces on AMReX's own communicator (ParallelContext::CommunicatorSub), the same one\n"
        "the solvers use.\n\n"
        "Collective: every rank must call it, with the same size. A no-op on one rank."
    );
}
