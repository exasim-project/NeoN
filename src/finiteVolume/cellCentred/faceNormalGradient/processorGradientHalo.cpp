// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#ifdef NF_WITH_MPI_SUPPORT

#include <cstddef>
#include <vector>

#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/processorGradientHalo.hpp"

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::detail
{

// Distinct tag from the geometry-scheme exchanges and BoundaryData::communicate (tag 0).
constexpr mpi_label_t procGradientHaloTag = 0x6772; // 'gr'

template<typename GradType>
Vector<GradType> exchangeProcNeighbourGradient(
    const Executor& exec, const UnstructuredMesh& mesh, const Vector<GradType>& gradInternal
)
{
    const auto nProcFaces = mesh.nProcBoundaryFaces();
    if (nProcFaces == 0) return Vector<GradType>(exec, 0, GradType {});

    const auto& bMesh = mesh.boundaryMesh();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    constexpr std::size_t nComp = sizeof(GradType) / sizeof(scalar);

    // Gather the owner-cell gradient for each processor face on the device.
    Vector<GradType> ownGradDev(exec, nProcFaces, GradType {});
    {
        auto ownView = ownGradDev.view();
        const auto gradView = gradInternal.view();
        const auto bFaceOwners = bMesh.faceOwners().view();
        parallelFor(
            exec,
            {0, nProcFaces},
            NEON_LAMBDA(const localIdx i) {
                ownView[i] = gradView[bFaceOwners[nBoundaryFaces + i]];
            },
            "exchangeProcNeighbourGradientGather"
        );
    }

    // Flatten to contiguous scalars for the typed MPI exchange.
    auto ownH = ownGradDev.copyToHost();
    const auto ownHView = ownH.view();
    std::vector<scalar> sendBuf(nComp * static_cast<std::size_t>(nProcFaces));
    for (localIdx i = 0; i < nProcFaces; ++i)
    {
        const scalar* d = ownHView[i].data();
        for (std::size_t c = 0; c < nComp; ++c)
            sendBuf[nComp * static_cast<std::size_t>(i) + c] = d[c];
    }
    std::vector<scalar> recvBuf(nComp * static_cast<std::size_t>(nProcFaces), scalar(0));

    // Processor patch face ranges (the trailing nProcBoundaryPatches patches of the boundary mesh).
    const auto& off = bMesh.offset();
    const auto nBounds = bMesh.nBoundaries();
    const auto nProcPatches = bMesh.nProcBoundaryPatches();
    std::vector<MPI_Request> requests(2 * static_cast<std::size_t>(nProcPatches), MPI_REQUEST_NULL);
    mpi::Environment mpiEnv;
    std::size_t p = 0;
    for (localIdx i = nBounds - nProcPatches; i < nBounds; ++i, ++p)
    {
        const auto rangeStart = off[static_cast<std::size_t>(i)];
        const auto rangeEnd = off[static_cast<std::size_t>(i + 1)];
        const auto patchOff = nComp * static_cast<std::size_t>(rangeStart - nBoundaryFaces);
        const auto neighborRank =
            static_cast<mpi_label_t>(bMesh.neighbourRankForRange({rangeStart, rangeEnd}));
        const auto count =
            static_cast<mpi_label_t>(nComp * static_cast<std::size_t>(rangeEnd - rangeStart));
        mpi::isend<scalar>(
            sendBuf.data() + patchOff,
            count,
            neighborRank,
            procGradientHaloTag,
            mpiEnv.comm(),
            &requests[2 * p]
        );
        mpi::irecv<scalar>(
            recvBuf.data() + patchOff,
            count,
            neighborRank,
            procGradientHaloTag,
            mpiEnv.comm(),
            &requests[2 * p + 1]
        );
    }
    mpi::waitAll(requests);

    std::vector<GradType> neiGrad(static_cast<std::size_t>(nProcFaces));
    for (localIdx i = 0; i < nProcFaces; ++i)
    {
        GradType g {};
        scalar* d = g.data();
        for (std::size_t c = 0; c < nComp; ++c)
            d[c] = recvBuf[nComp * static_cast<std::size_t>(i) + c];
        neiGrad[static_cast<std::size_t>(i)] = g;
    }
    return Vector<GradType>(exec, neiGrad);
}

// Explicit instantiations: Vec3 (gradient of a scalar field) and Tensor (gradient of a
// Vec3 field) are the only gradient types produced by the face-normal-gradient schemes.
template Vector<Vec3> exchangeProcNeighbourGradient<Vec3>(
    const Executor& exec, const UnstructuredMesh& mesh, const Vector<Vec3>& gradInternal
);
template Vector<Tensor> exchangeProcNeighbourGradient<Tensor>(
    const Executor& exec, const UnstructuredMesh& mesh, const Vector<Tensor>& gradInternal
);

} // namespace NeoN::finiteVolume::cellCentred::detail

#endif
