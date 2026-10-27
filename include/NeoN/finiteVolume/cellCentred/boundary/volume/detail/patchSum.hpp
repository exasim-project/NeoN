// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <Kokkos_Core.hpp>

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"
#endif

namespace NeoN::finiteVolume::cellCentred::volumeBoundary::detail
{

/**
 * @brief Sum of a per-face quantity over the faces of one patch, over all ranks.
 *
 * The counterpart of OpenFOAM's gSum on a patch field, as used by boundary conditions that
 * distribute a prescribed quantity over the whole patch (fixedMean, flowRateInletVelocity).
 * The kernel has the signature `(const localIdx i, scalar& localSum)` and is evaluated for every
 * boundary face index i of `range`; the rank-local sum is then reduced over all ranks when the
 * mesh is distributed. Every rank has to call this, also ranks that hold no face of the patch.
 *
 * @param mesh The mesh the patch belongs to.
 * @param range The boundary face range of the patch.
 * @param kernel Adds the contribution of face i to localSum.
 * @return The sum over the patch faces of all ranks.
 */
template<typename Kernel>
scalar
patchSum(const UnstructuredMesh& mesh, std::pair<localIdx, localIdx> range, const Kernel& kernel)
{
    scalar sum = scalar(0);
    Kokkos::Sum<scalar> reducer(sum);
    NeoN::parallelReduce(mesh.exec(), range, kernel, reducer);

#ifdef NF_WITH_MPI_SUPPORT
    if (mesh.boundaryMesh().isDistributed())
    {
        mpi::Environment env;
        mpi::allReduce(sum, mpi::ReduceOp::Sum, env.comm());
    }
#endif

    return sum;
}

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary::detail
