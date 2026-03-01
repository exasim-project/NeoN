// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN
{

UnstructuredMesh
createUniform2DGrid(const Executor exec, localIdx nx, localIdx ny, scalar Lx, scalar Ly)
{
    return createUniform3DGrid(exec, nx, ny, 1, Lx, Ly, 1.0);
}

} // namespace NeoN
