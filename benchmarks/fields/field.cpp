// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

std::vector<NeoN::localIdx> create3PatchOffsets(NeoN::localIdx nBoundaryFaces)
{
    // Create patches - {0, 1/3, 2/3, 1}
    // If nBoundaryFaces not divisible by 3, then the extras go the final patch
    return {0, nBoundaryFaces / 3, (2 * nBoundaryFaces) / 3, nBoundaryFaces};
}

TEMPLATE_TEST_CASE(
    "Field::1D", "[bench]", NeoN::Vec3
) //"Template" for consistency with other benchmarks.
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const NeoN::localIdx nCells = static_cast<NeoN::localIdx>(size);

    // nBnd should be such that internal/boundary ratio is maintained as size varies
    const NeoN::localIdx nBnd = std::max<NeoN::localIdx>(NeoN::localIdx(128), nCells / 16);
    const auto offsets = create3PatchOffsets(nBnd);

    NeoN::Field<TestType> field(exec, nCells, offsets);

    const std::string sectionName = std::to_string(nCells);

    DYNAMIC_SECTION(sectionName + " - Internal_fill")
    {
        NeoN::fill(field.internalVector(), NeoN::one<TestType>());

        BENCHMARK(std::string(execName))
        {
            NeoN::fill(field.internalVector(), 2.0 * NeoN::one<TestType>());
        };
    }

    DYNAMIC_SECTION(sectionName + " - Internal_copy")
    {
        // Pre-allocate both fields once; benchmark measures copy only
        NeoN::Field<TestType> src(exec, nCells, offsets);
        NeoN::Field<TestType> dst(exec, nCells, offsets);
        NeoN::fill(src.internalVector(), 3.0 * NeoN::one<TestType>());
        NeoN::fill(dst.internalVector(), NeoN::zero<TestType>());

        BENCHMARK(std::string(execName)) { dst.internalVector() = src.internalVector(); };
    }

    DYNAMIC_SECTION(sectionName + " - Boundary_value_fill")
    {
        NeoN::fill(field.boundaryData().value(), NeoN::one<TestType>());

        BENCHMARK(std::string(execName))
        {
            NeoN::fill(field.boundaryData().value(), 2.0 * NeoN::one<TestType>());
        };
    }

    DYNAMIC_SECTION(sectionName + " - Boundary_value_copy")
    {
        // Pre-allocate both fields once; benchmark measures copy only
        NeoN::Field<TestType> src(exec, nCells, offsets);
        NeoN::Field<TestType> dst(exec, nCells, offsets);
        NeoN::fill(src.boundaryData().value(), 4.0 * NeoN::one<TestType>());
        NeoN::fill(dst.boundaryData().value(), NeoN::zero<TestType>());

        BENCHMARK(std::string(execName))
        {
            dst.boundaryData().value() = src.boundaryData().value();
        };
    }
}
