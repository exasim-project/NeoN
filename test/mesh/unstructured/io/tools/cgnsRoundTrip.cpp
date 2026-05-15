// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>

#include "NeoN/core/initialization.hpp"
#include "NeoN/core/executor/serialExecutor.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshReader.hpp"
#include "NeoN/mesh/unstructured/io/cgnsMeshWriter.hpp"

int main(int argc, char* argv[])
{
    NeoN::initialize(argc, argv);

    if (argc < 3)
    {
        std::cerr << "Usage: cgnsRoundTrip <input.cgns> <output.cgns>\n";
        NeoN::finalize();
        return 1;
    }

    {
        NeoN::SerialExecutor exec;
        auto mesh = NeoN::io::readCgns(argv[1], exec);
        NeoN::io::writeCgns(mesh, argv[2]);

        std::cout << "cells=" << mesh.nCells() << " faces=" << mesh.nFaces()
                  << " boundaries=" << mesh.nBoundaries() << "\n";
    }

    NeoN::finalize();
    return 0;
}
