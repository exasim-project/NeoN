// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef NF_WITH_MPI_SUPPORT

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/fields/field.hpp"

namespace NeoN
{
class UnstructuredMesh;
}

namespace NeoN::finiteVolume::cellCentred::detail
{

/** @brief For every processor-boundary face, returns the gradient of the cell on the far side of
 *  the rank boundary (grad_nei), obtained by exchanging owner-cell gradients with neighbouring
 *  ranks. The result is a device Vector of size nProcBoundaryFaces, indexed by processor face.
 *
 *  Instantiated for GradType = Vec3 (gradient of a scalar field) and GradType = Tensor
 *  (gradient of a Vec3 field). */
template<typename GradType>
Vector<GradType> exchangeProcNeighbourGradient(
    const Executor& exec, const UnstructuredMesh& mesh, const Vector<GradType>& gradInternal
);

} // namespace NeoN::finiteVolume::cellCentred::detail

#endif
