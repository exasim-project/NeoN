// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <iostream>

#include "catch2/catch_session.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators_adapters.hpp"
#include "catch2/reporters/catch_reporter_registrars.hpp"

#include <Kokkos_Core.hpp>
#include "NeoN/NeoN.hpp"

int main(int argc, char* argv[])
{
    // Initialize Catch2
    NeoN::initialize(argc, argv);

    // ensure any kokkos initialization output will appear first
    std::cout << std::flush;
    std::cerr << std::flush;
    int result;
    {
        Catch::Session session;

        // Specify command line options
        int returnCode = session.applyCommandLine(argc, argv);
        if (returnCode != 0) // Indicates a command line error
            return returnCode;
        result = session.run();
    }
    NeoN::finalize();

    return result;
}
