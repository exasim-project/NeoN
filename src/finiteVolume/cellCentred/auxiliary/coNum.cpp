// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <limits>

#include "NeoN/core/info.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/auxiliary/coNum.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/core/mpi/operators.hpp"
#endif

namespace NeoN::finiteVolume::cellCentred
{

std::pair<scalar, scalar> computeCoNum(const SurfaceField<scalar>& faceFlux, const scalar dt)
{
    const UnstructuredMesh& mesh = faceFlux.mesh();
    const auto exec = faceFlux.exec();
    VolumeField<scalar> phi(exec, "phi", mesh, createCalculatedBCs<VolumeBoundary<scalar>>(mesh));
    fill(phi.internalVector(), 0.0);

    const auto [surfFaceCells, volPhi, surfOwner, surfNeighbour, surfFaceFlux, surfV] = views(
        mesh.boundaryMesh().faceCells(),
        phi.internalVector(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        faceFlux.internalVector(),
        mesh.cellVolumes()
    );
    auto nInternalFaces = mesh.nInternalFaces();

    scalar maxCoNum = std::numeric_limits<scalar>::lowest();
    scalar meanCoNum = 0.0;
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            scalar flux = Kokkos::sqrt(surfFaceFlux[i] * surfFaceFlux[i]);
            Kokkos::atomic_add(&volPhi[surfOwner[i]], flux);
            Kokkos::atomic_add(&volPhi[surfNeighbour[i]], flux);
        },
        "computeCoNum::fluxInternal"
    );

    parallelFor(
        exec,
        {nInternalFaces, faceFlux.size()},
        NEON_LAMBDA(const localIdx i) {
            auto own = surfFaceCells[i - nInternalFaces];
            scalar flux = Kokkos::sqrt(surfFaceFlux[i] * surfFaceFlux[i]);
            Kokkos::atomic_add(&volPhi[own], flux);
        },
        "computeCoNum::fluxBoundary"
    );

    phi.correctBoundaryConditions();

    scalar maxValue {0.0};
    Kokkos::Max<NeoN::scalar> maxReducer(maxValue);
    parallelReduce(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx celli, NeoN::scalar& lmax) {
            NeoN::scalar val = (volPhi[celli] / surfV[celli]);
            if (val > lmax) lmax = val;
        },
        maxReducer
    );

    scalar totalPhi = 0.0;
    Kokkos::Sum<NeoN::scalar> sumPhi(totalPhi);
    parallelReduce(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx celli, scalar& lsum) { lsum += volPhi[celli]; },
        sumPhi
    );

    scalar totalVol = 0.0;
    Kokkos::Sum<NeoN::scalar> sumVol(totalVol);
    parallelReduce(
        exec,
        {0, mesh.nCells()},
        NEON_LAMBDA(const localIdx celli, scalar& lsum) { lsum += surfV[celli]; },
        sumVol
    );

#ifdef NF_WITH_MPI_SUPPORT
    if (mesh.boundaryMesh().isDistributed())
    {
        mpi::Environment env;
        mpi::allReduce(maxValue, mpi::ReduceOp::Max, env.comm());
        scalar sums[2] = {totalPhi, totalVol};
        MPI_Allreduce(MPI_IN_PLACE, sums, 2, mpi::getType<scalar>(), MPI_SUM, env.comm());
        totalPhi = sums[0];
        totalVol = sums[1];
    }
#endif

    maxCoNum = maxValue * 0.5 * dt;
    meanCoNum = 0.5 * (totalPhi / totalVol) * dt;

    return {maxCoNum, meanCoNum};
}

};
