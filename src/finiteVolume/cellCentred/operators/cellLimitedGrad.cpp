// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/cellLimitedGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/view.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @brief minmod limiter ratio for one cell-face pair.
 *
 * Returns min(r, 1), where r bounds the face-extrapolated value within the
 * cell's admissible delta range [minDelta, maxDelta]. A negligible extrapolation
 * imposes no constraint and returns 1. With maxDelta >= 0 and minDelta <= 0 the
 * result is always within [0, 1].
 */
NEON_INLINE_FUNCTION
scalar cellLimiterRatio(const scalar extrapolate, const scalar maxDelta, const scalar minDelta)
{
    if (extrapolate > ROOTVSMALL)
    {
        return Kokkos::min(maxDelta / extrapolate, scalar(1));
    }
    if (extrapolate < -ROOTVSMALL)
    {
        return Kokkos::min(minDelta / extrapolate, scalar(1));
    }
    return scalar(1);
}

/* @brief free-standing implementation of the cell-limited explicit gradient.
 *
 * Computes the unlimited base gradient, derives a per-cell scalar limiter that
 * clips the gradient so cell-to-face extrapolated values stay bounded by the
 * neighbour cell values, then applies the limiter and the operator scaling.
 */
void computeCellLimitedGrad(
    const VolumeField<scalar>& phi,
    const GradOperatorFactory<Vec3>& baseScheme,
    const GeometryScheme& geometryScheme,
    const scalar coeff,
    const dsl::Coeff operatorScaling,
    Vector<Vec3>& gradPhi
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = gradPhi.exec();
    const auto nCells = mesh.nCells();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();

    // 1. Unlimited base gradient. The limiter is a geometric property of the
    //    field, so the base gradient is computed without operator scaling and the
    //    scaling is folded in only at the final step.
    baseScheme.grad(phi, dsl::Coeff(scalar(1)), gradPhi);

    auto g = gradPhi.view();
    const auto phiV = phi.internalVector().view();

    // coeff == 0: limiting disabled — apply operator scaling and return.
    if (coeff < ROOTVSMALL)
    {
        parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx celli) { g[celli] *= operatorScaling[celli]; },
            "cellLimitedGradNoLimit"
        );
        return;
    }

    // 2. Per-cell admissible value range: max/min over the cell value and all
    //    face-neighbour cell values.
    Vector<scalar> maxVsf(exec, nCells, scalar(0));
    Vector<scalar> minVsf(exec, nCells, scalar(0));
    auto maxV = maxVsf.view();
    auto minV = minVsf.view();
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            maxV[celli] = phiV[celli];
            minV[celli] = phiV[celli];
        },
        "cellLimitedInitRange"
    );

    const auto [owner, neighbour, bOwner] =
        views(mesh.faceOwners(), mesh.faceNeighbors(), mesh.boundaryMesh().faceOwners());

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto own = owner[facei];
            const auto nei = neighbour[facei];
            const scalar vOwn = phiV[own];
            const scalar vNei = phiV[nei];
            Kokkos::atomic_max(&maxV[own], vNei);
            Kokkos::atomic_min(&minV[own], vNei);
            Kokkos::atomic_max(&maxV[nei], vOwn);
            Kokkos::atomic_min(&minV[nei], vOwn);
        },
        "cellLimitedRangeInternal"
    );

    const auto phiBValue = phi.boundaryData().value().view();
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            const auto own = bOwner[bfi];
            const scalar vb = phiBValue[bfi];
            Kokkos::atomic_max(&maxV[own], vb);
            Kokkos::atomic_min(&minV[own], vb);
        },
        "cellLimitedRangeBoundary"
    );

    // Processor-boundary faces: fold the communicated neighbour value into the
    // owner range. Mirrors the physical-boundary loop so the limiter extends to
    // distributed runs; the slot nBoundaryFaces+procFacei carries the halo value.
    if (nProcBoundaryFaces > 0)
    {
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const auto bfi = nBoundaryFaces + procFacei;
                const auto own = bOwner[bfi];
                const scalar vb = phiBValue[bfi];
                Kokkos::atomic_max(&maxV[own], vb);
                Kokkos::atomic_min(&minV[own], vb);
            },
            "cellLimitedRangeProcBoundary"
        );
    }

    // 3. Convert the absolute range to deltas about the cell value and widen the
    //    bounds by the coefficient (k < 1 relaxes the limiter).
    const scalar k = coeff;
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            scalar mx = maxV[celli] - phiV[celli];
            scalar mn = minV[celli] - phiV[celli];
            if (k < scalar(1))
            {
                const scalar d = (scalar(1) / k - scalar(1)) * (mx - mn);
                mx += d;
                mn -= d;
            }
            maxV[celli] = mx;
            minV[celli] = mn;
        },
        "cellLimitedDeltas"
    );

    // 4. Limiter: clip each cell's gradient so the value extrapolated to every
    //    surrounding face stays within the admissible delta range.
    Vector<scalar> limiter(exec, nCells, scalar(1));
    auto lim = limiter.view();

    // Internal-face cell-to-face displacement vectors come from the geometry scheme:
    // the raw mesh cell/face centres are freed once the geometry scheme is built, so
    // they cannot be read here directly.
    const auto ownerToFace = geometryScheme.faceDeltaOwner().internalVector().view();
    const auto neighbourToFace = geometryScheme.faceDeltaNeighbour().internalVector().view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto own = owner[facei];
            const auto nei = neighbour[facei];
            const scalar extrapOwn = ownerToFace[facei] & g[own];
            const scalar extrapNei = neighbourToFace[facei] & g[nei];
            Kokkos::atomic_min(&lim[own], cellLimiterRatio(extrapOwn, maxV[own], minV[own]));
            Kokkos::atomic_min(&lim[nei], cellLimiterRatio(extrapNei, maxV[nei], minV[nei]));
        },
        "cellLimitedLimiterInternal"
    );

    const auto [bFaceC, bOwnerC] =
        views(mesh.boundaryMesh().faceCenters(), mesh.boundaryMesh().ownerCellCenters());
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            const auto own = bOwner[bfi];
            const scalar extrap = (bFaceC[bfi] - bOwnerC[bfi]) & g[own];
            Kokkos::atomic_min(&lim[own], cellLimiterRatio(extrap, maxV[own], minV[own]));
        },
        "cellLimitedLimiterBoundary"
    );

    if (nProcBoundaryFaces > 0)
    {
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const auto bfi = nBoundaryFaces + procFacei;
                const auto own = bOwner[bfi];
                const scalar extrap = (bFaceC[bfi] - bOwnerC[bfi]) & g[own];
                Kokkos::atomic_min(&lim[own], cellLimiterRatio(extrap, maxV[own], minV[own]));
            },
            "cellLimitedLimiterProcBoundary"
        );
    }

    // 5. Apply the per-cell limiter and the operator scaling to the gradient.
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) { g[celli] *= (operatorScaling[celli] * lim[celli]); },
        "cellLimitedApply"
    );
}

CellLimitedGrad::CellLimitedGrad(
    const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs
)
    : Base(exec, mesh), baseGradScheme_(GradOperatorFactory<Vec3>::create(exec, mesh, inputs)),
      coeff_(readCoeff(inputs)), geometryScheme_(GeometryScheme::readOrCreate(mesh))
{
    if (!(coeff_ >= scalar(0) && coeff_ <= scalar(1)))
    {
        NF_THROW(
            std::string("cellLimited gradient coefficient must be in [0, 1], got ")
            + std::to_string(coeff_)
        );
    }
}

scalar CellLimitedGrad::readCoeff(const Input& inputs)
{
    if (std::holds_alternative<Dictionary>(inputs))
    {
        const auto& dict = std::get<Dictionary>(inputs);
        if (dict.contains("cellLimitedCoeff"))
        {
            return dict.get<scalar>("cellLimitedCoeff");
        }
        return DEFAULT_COEFF;
    }

    // TokenList form "cellLimited <baseScheme ...> <coeff>": the factory consumed
    // "cellLimited" and the base scheme consumed its own tokens (e.g. "Gauss
    // linear"); only the trailing coefficient remains. Skip any unconsumed
    // sub-scheme words, then read the coefficient — written as an integer (e.g. 1)
    // or a fractional value (e.g. 0.5).
    const auto& tl = std::get<TokenList>(inputs);
    while (tl.peekIs<std::string>())
    {
        tl.next<std::string>();
    }
    if (tl.peekIs<scalar>())
    {
        return tl.next<scalar>();
    }
    if (tl.peekIs<label>())
    {
        return static_cast<scalar>(tl.next<label>());
    }
    return DEFAULT_COEFF;
}

void CellLimitedGrad::grad(
    const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling, Vector<Vec3>& gradPhi
) const
{
    computeCellLimitedGrad(
        phi, *baseGradScheme_, *geometryScheme_, coeff_, operatorScaling, gradPhi
    );
}

VolumeField<Vec3>
CellLimitedGrad::grad(const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling) const
{
    // Proc-aware calculated BCs mirror GaussGreenGrad::grad so processor patches
    // carry the halo-exchange BC for the boundary gradient refresh below.
    auto gradBCs = createCalculatedProcBCs<VolumeBoundary<Vec3>>(phi.mesh());
    VolumeField<Vec3> gradPhi(phi.exec(), "gradPhi", phi.mesh(), gradBCs);
    fill(gradPhi.internalVector(), zero<Vec3>());
    computeCellLimitedGrad(
        phi, *baseGradScheme_, *geometryScheme_, coeff_, operatorScaling, gradPhi.internalVector()
    );
    gradPhi.correctBoundaryConditions();
    return gradPhi;
}

} // namespace NeoN::finiteVolume::cellCentred
