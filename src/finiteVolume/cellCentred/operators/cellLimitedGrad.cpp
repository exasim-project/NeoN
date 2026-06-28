// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/cellLimitedGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/cellToFaceStencil.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/view.hpp"

#include <Kokkos_Profiling_ScopedRegion.hpp> // cell-limited-grad phase regions (no-op without a tool)

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
      coeff_(readCoeff(inputs)), geometryScheme_(GeometryScheme::readOrCreate(mesh)),
      cellInternalFaces_(CellToFaceStencil(mesh).computeInternalStencil())
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

/* @brief GPU-first cell-limited tensor gradient of a vector field.
 *
 * Limits grad(U) per component: three independent minmod limiters (one per U
 * component), clipping the gradient so the value extrapolated from the cell centre
 * to each surrounding face stays within the neighbour cell-value range. Row i of
 * gradU (= grad of U_i) is scaled by limiter component i.
 *
 * Iteration is cell-based: each cell gathers its own internal-face stencil in
 * registers (no atomics, coalesced writes); only the comparatively small set of
 * physical/processor boundary faces uses an atomic scatter into the owner cell.
 */
void computeCellLimitedGradTensor(
    const VolumeField<Vec3>& u,
    const GradOperatorFactory<Vec3>& baseScheme,
    const GeometryScheme& geometryScheme,
    const SegmentedVector<localIdx, localIdx>& cellInternalFaces,
    const scalar coeff,
    const dsl::Coeff operatorScaling,
    VolumeField<Tensor>& gradU
)
{
    const UnstructuredMesh& mesh = u.mesh();
    const auto exec = gradU.exec();
    const auto nCells = mesh.nCells();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();

    // 1. Unlimited base tensor gradient (internal + boundary), without operator scaling. This is a
    // full GaussGreenGrad (grad + boundary reconstruction) -- profiled apart from the limiter so we
    // can tell the base-gradient cost from the min/max-range + limiter machinery below.
    {
        Kokkos::Profiling::ScopedRegion region_("clg.baseGrad");
        baseScheme.gradTensor(u, gradU, dsl::Coeff(scalar(1)));
    }

    auto g = gradU.internalVector().view();
    const auto uV = u.internalVector().view();

    // coeff == 0: limiting disabled — apply operator scaling and return.
    if (coeff < ROOTVSMALL)
    {
        parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx celli) { g[celli] *= operatorScaling[celli]; },
            "cellLimitedGradTensorNoLimit"
        );
        return;
    }

    // The minmod limiter machinery: per-component neighbour-range gather, limiter ratios, and the
    // final apply. Region declared AFTER the early return so it stays push/pop-balanced; it covers
    // the remainder of the function (no further return).
    Kokkos::Profiling::ScopedRegion limiterRegion_("clg.limiter");

    // Per cell, per component (Vec3 = one scalar per U component).
    Vector<Vec3> maxDelta(exec, nCells, zero<Vec3>());
    Vector<Vec3> minDelta(exec, nCells, zero<Vec3>());
    Vector<Vec3> limiter(exec, nCells, one<Vec3>());
    auto maxD = maxDelta.view();
    auto minD = minDelta.view();
    auto lim = limiter.view();

    const auto [owner, neighbour, bOwner] =
        views(mesh.faceOwners(), mesh.faceNeighbors(), mesh.boundaryMesh().faceOwners());
    const auto cellFaces = cellInternalFaces.values().view();
    const auto cellFaceOffs = cellInternalFaces.segments().view();

    // 2a. Cell-based gather (atomic-free): component-wise neighbour deltas over internal faces.
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            const Vec3 uOwn = uV[celli];
            Vec3 mx = zero<Vec3>();
            Vec3 mn = zero<Vec3>();
            for (auto sIdx = cellFaceOffs[celli]; sIdx < cellFaceOffs[celli + 1]; ++sIdx)
            {
                const auto facei = cellFaces[sIdx];
                const auto other = (owner[facei] == celli) ? neighbour[facei] : owner[facei];
                const Vec3 dU = uV[other] - uOwn;
                for (size_t c = 0; c < 3; ++c)
                {
                    mx[c] = Kokkos::max(mx[c], dU[c]);
                    mn[c] = Kokkos::min(mn[c], dU[c]);
                }
            }
            maxD[celli] = mx;
            minD[celli] = mn;
        },
        "cellLimitedTensorRangeInternal"
    );

    // 2b. Boundary scatter (atomics, small): fold boundary-face values into the owner range.
    const auto uBoundary = u.boundaryData().value().view();
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            const auto own = bOwner[bfi];
            const Vec3 dU = uBoundary[bfi] - uV[own];
            for (size_t c = 0; c < 3; ++c)
            {
                Kokkos::atomic_max(&maxD[own][c], dU[c]);
                Kokkos::atomic_min(&minD[own][c], dU[c]);
            }
        },
        "cellLimitedTensorRangeBoundary"
    );

    if (nProcBoundaryFaces > 0)
    {
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const auto bfi = nBoundaryFaces + procFacei;
                const auto own = bOwner[bfi];
                const Vec3 dU = uBoundary[bfi] - uV[own];
                for (size_t c = 0; c < 3; ++c)
                {
                    Kokkos::atomic_max(&maxD[own][c], dU[c]);
                    Kokkos::atomic_min(&minD[own][c], dU[c]);
                }
            },
            "cellLimitedTensorRangeProcBoundary"
        );
    }

    // 3. Widen the per-component bounds by the coefficient (k < 1 relaxes the limiter).
    const scalar k = coeff;
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            Vec3 mx = maxD[celli];
            Vec3 mn = minD[celli];
            if (k < scalar(1))
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    const scalar d = (scalar(1) / k - scalar(1)) * (mx[c] - mn[c]);
                    mx[c] += d;
                    mn[c] -= d;
                }
            }
            maxD[celli] = mx;
            minD[celli] = mn;
        },
        "cellLimitedTensorDeltas"
    );

    // 4a. Cell-based gather (atomic-free): per-component limiter over internal faces. The
    //     base (unlimited) gradient extrapolated to each face must stay within the deltas.
    const auto ownerToFace = geometryScheme.faceDeltaOwner().internalVector().view();
    const auto neighbourToFace = geometryScheme.faceDeltaNeighbour().internalVector().view();
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            const Tensor gc = g[celli];
            const Vec3 mx = maxD[celli];
            const Vec3 mn = minD[celli];
            Vec3 l = one<Vec3>();
            for (auto sIdx = cellFaceOffs[celli]; sIdx < cellFaceOffs[celli + 1]; ++sIdx)
            {
                const auto facei = cellFaces[sIdx];
                const Vec3 d =
                    (owner[facei] == celli) ? ownerToFace[facei] : neighbourToFace[facei];
                const Vec3 extrap = gc & d; // predicted delta of each U component at the face
                for (size_t c = 0; c < 3; ++c)
                {
                    l[c] = Kokkos::min(l[c], cellLimiterRatio(extrap[c], mx[c], mn[c]));
                }
            }
            lim[celli] = l;
        },
        "cellLimitedTensorLimiterInternal"
    );

    // 4b. Boundary scatter (atomics, small): per-component limiter over boundary faces.
    const auto [bFaceC, bOwnerC] =
        views(mesh.boundaryMesh().faceCenters(), mesh.boundaryMesh().ownerCellCenters());
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            const auto own = bOwner[bfi];
            const Vec3 extrap = g[own] & (bFaceC[bfi] - bOwnerC[bfi]);
            for (size_t c = 0; c < 3; ++c)
            {
                Kokkos::atomic_min(
                    &lim[own][c], cellLimiterRatio(extrap[c], maxD[own][c], minD[own][c])
                );
            }
        },
        "cellLimitedTensorLimiterBoundary"
    );

    if (nProcBoundaryFaces > 0)
    {
        parallelFor(
            exec,
            {0, nProcBoundaryFaces},
            NEON_LAMBDA(const localIdx procFacei) {
                const auto bfi = nBoundaryFaces + procFacei;
                const auto own = bOwner[bfi];
                const Vec3 extrap = g[own] & (bFaceC[bfi] - bOwnerC[bfi]);
                for (size_t c = 0; c < 3; ++c)
                {
                    Kokkos::atomic_min(
                        &lim[own][c], cellLimiterRatio(extrap[c], maxD[own][c], minD[own][c])
                    );
                }
            },
            "cellLimitedTensorLimiterProcBoundary"
        );
    }

    // 5. Apply: scale row i of gradU (grad of U_i) by limiter[i], then the operator scaling.
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            const Vec3 l = lim[celli];
            const scalar s = operatorScaling[celli];
            Tensor gc = g[celli];
            for (size_t i = 0; i < 3; ++i)
            {
                const scalar si = l[i] * s;
                for (size_t j = 0; j < 3; ++j)
                {
                    gc(i, j) *= si;
                }
            }
            g[celli] = gc;
        },
        "cellLimitedTensorApply"
    );
}

void CellLimitedGrad::gradTensor(
    const VolumeField<Vec3>& u, VolumeField<Tensor>& gradU, const dsl::Coeff operatorScaling
) const
{
    computeCellLimitedGradTensor(
        u, *baseGradScheme_, *geometryScheme_, cellInternalFaces_, coeff_, operatorScaling, gradU
    );
}

} // namespace NeoN::finiteVolume::cellCentred
