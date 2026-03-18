// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include <AMReX.H>

namespace nb = nanobind;

void registerInit(nb::module_& m)
{
    m.def("initialize", []() {
        int argc = 0;
        char** argv = nullptr;
        amrex::Initialize(argc, argv);
    });
    m.def("finalize", []() { amrex::Finalize(); });
}
