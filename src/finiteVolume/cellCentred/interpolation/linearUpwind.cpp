// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>

#include "NeoN/finiteVolume/cellCentred/interpolation/linearUpwind.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<>
void computeLinearUpwindInterpolation<scalar>(
    const VolumeField<scalar>& src,
    const SurfaceField<scalar>& flux,
    const VolumeField<Vec3>& gradPhi,
    const vectorVector& faceCentres,
    const vectorVector& cellCentres,
    const UnstructuredMesh& mesh,
    SurfaceField<scalar>& dst
)
{
    const auto exec = dst.exec();
    auto dstS = dst.internalVector().view();
    const auto [srcS, fluxS, gradS, ownerS, neighS, boundS, faceCentresS, cellCentresS] = views(
        src.internalVector(),
        flux.internalVector(),
        gradPhi.internalVector(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        src.boundaryData().value(),
        faceCentres,
        cellCentres
    );
    auto nInternalFaces = mesh.nInternalFaces();

    parallelFor(
        exec,
        {0, dstS.size()},
        NEON_LAMBDA(const localIdx facei) {
            if (facei < nInternalFaces)
            {
                auto own = ownerS[facei];
                auto nei = neighS[facei];
                scalar phif;
                if (fluxS[facei] >= 0)
                {
                    Vec3 d = faceCentresS[facei] - cellCentresS[own];
                    phif = srcS[own] + (d & gradS[own]);
                }
                else
                {
                    Vec3 d = faceCentresS[facei] - cellCentresS[nei];
                    phif = srcS[nei] + (d & gradS[nei]);
                }
                // Limit between owner and neighbour values
                scalar maxVal = srcS[own] > srcS[nei] ? srcS[own] : srcS[nei];
                scalar minVal = srcS[own] < srcS[nei] ? srcS[own] : srcS[nei];
                phif = phif > maxVal ? maxVal : phif;
                phif = phif < minVal ? minVal : phif;
                dstS[facei] = phif;
            }
            else
            {
                dstS[facei] = boundS[facei - nInternalFaces];
            }
        },
        "computeLinearUpwindInterpolation_scalar"
    );
}

void computeLinearUpwindInterpolation(
    const VolumeField<Vec3>& src,
    const SurfaceField<scalar>& flux,
    const TensorVecField& gradU,
    const vectorVector& faceCentres,
    const vectorVector& cellCentres,
    const UnstructuredMesh& mesh,
    SurfaceField<Vec3>& dst
)
{
    const auto exec = dst.exec();
    auto dstS = dst.internalVector().view();
    const auto
        [srcS, fluxS, gradUxS, gradUyS, gradUzS, ownerS, neighS, boundS, faceCentresS,
         cellCentresS] =
            views(
                src.internalVector(),
                flux.internalVector(),
                gradU.Tx.internalVector(),
                gradU.Ty.internalVector(),
                gradU.Tz.internalVector(),
                mesh.faceOwner(),
                mesh.faceNeighbour(),
                src.boundaryData().value(),
                faceCentres,
                cellCentres
            );
    auto nInternalFaces = mesh.nInternalFaces();

    parallelFor(
        exec,
        {0, dstS.size()},
        NEON_LAMBDA(const localIdx facei) {
            if (facei < nInternalFaces)
            {
                auto own = ownerS[facei];
                auto nei = neighS[facei];
                Vec3 phif;
                if (fluxS[facei] >= 0)
                {
                    Vec3 d = faceCentresS[facei] - cellCentresS[own];
                    phif[0] = srcS[own][0] + (d & gradUxS[own]);
                    phif[1] = srcS[own][1] + (d & gradUyS[own]);
                    phif[2] = srcS[own][2] + (d & gradUzS[own]);
                }
                else
                {
                    Vec3 d = faceCentresS[facei] - cellCentresS[nei];
                    phif[0] = srcS[nei][0] + (d & gradUxS[nei]);
                    phif[1] = srcS[nei][1] + (d & gradUyS[nei]);
                    phif[2] = srcS[nei][2] + (d & gradUzS[nei]);
                }
                // Limit each component between owner and neighbour values
                for (localIdx k = 0; k < 3; ++k)
                {
                    scalar maxVal = srcS[own][k] > srcS[nei][k] ? srcS[own][k] : srcS[nei][k];
                    scalar minVal = srcS[own][k] < srcS[nei][k] ? srcS[own][k] : srcS[nei][k];
                    phif[k] = phif[k] > maxVal ? maxVal : phif[k];
                    phif[k] = phif[k] < minVal ? minVal : phif[k];
                }
                dstS[facei] = phif;
            }
            else
            {
                dstS[facei] = boundS[facei - nInternalFaces];
            }
        },
        "computeLinearUpwindInterpolation_Vec3"
    );
}

template<>
void LinearUpwind<scalar>::interpolate(
    const SurfaceField<scalar>& flux,
    const VolumeField<scalar>& src,
    SurfaceField<scalar>& dst
) const
{
    auto gradPhi = gaussGreenGrad_.grad(src);
    computeLinearUpwindInterpolation<scalar>(
        src, flux, gradPhi, faceCentres_, cellCentres_, this->mesh_, dst
    );
}

template<>
void LinearUpwind<Vec3>::interpolate(
    const SurfaceField<scalar>& flux,
    const VolumeField<Vec3>& src,
    SurfaceField<Vec3>& dst
) const
{
    auto gradU = gaussGreenGrad_.grad(src);
    computeLinearUpwindInterpolation(
        src, flux, gradU, faceCentres_, cellCentres_, this->mesh_, dst
    );
}

} // namespace NeoN
