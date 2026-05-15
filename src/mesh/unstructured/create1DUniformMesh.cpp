// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN
{

UnstructuredMesh create1DUniformMesh(const Executor exec, const localIdx nCells)
{
    return createUniform3DGrid(exec, nCells, 1, 1);
}

} // namespace NeoN
