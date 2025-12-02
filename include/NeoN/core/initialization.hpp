// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/logging.hpp"

#include <Kokkos_Core.hpp>
#include <chrono>


namespace NeoN
{

inline void initialize(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    Logging::setNeonDefaultPattern();
}

inline void finalize()
{
    Logging::info("Finalizing NeoN");
    Kokkos::finalize();
}
}
