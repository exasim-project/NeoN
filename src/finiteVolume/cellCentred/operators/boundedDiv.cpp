// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/boundedDiv.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/traits.hpp"

namespace NeoN::finiteVolume::cellCentred
{

namespace
{

// diag[i] -= Σ_f φ_f * scaling[i] over internal faces.
//
// Sign convention: φ_f > 0 means flow from owner to neighbor, so each
// internal face contributes +φ_f to the owner cell's net outflow and -φ_f
// to the neighbor cell's. The bounded correction subtracts that outflow
// from the augmented diagonal.
template<typename FieldValueType, typename AssemblyType>
void applyBoundedDiagInternal(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const UnstructuredMesh& mesh,
    const dsl::Coeff scaling
)
{
    const auto exec = ls.exec();
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    const auto [fluxV, ownV, neiV] =
        views(faceFlux.internalVector(), mesh.faceOwners(), mesh.faceNeighbors());
    auto values = ls.matrix().values().view();
    const auto nInternalFaces = mesh.nInternalFaces();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            const auto ownRow = ownV[facei];
            const auto neiRow = neiV[facei];
            const auto phi_f = fluxV[facei];

            // diag[O] -= φ_f * scaling[O]
            Kokkos::atomic_sub(
                &values[ma.diagIdx(ownRow)], phi_f * scaling[ownRow] * one<AssemblyType>()
            );
            // diag[N] += φ_f * scaling[N]  (φ_f is -outflow for N)
            Kokkos::atomic_add(
                &values[ma.diagIdx(neiRow)], phi_f * scaling[neiRow] * one<AssemblyType>()
            );
        },
        "BoundedDiv::applyBoundedDiagInternal"
    );
}

// diag[i] -= φ_f * scaling[i] over regular boundary faces. Boundary face
// normals point outward from the local domain, so the boundary flux value
// IS the outflow contribution for the owner cell — no sign flip needed.
template<typename FieldValueType, typename AssemblyType>
void applyBoundedDiagBoundary(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const UnstructuredMesh& mesh,
    const dsl::Coeff scaling
)
{
    const auto exec = ls.exec();
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    const auto ownV = mesh.boundaryMesh().faceOwners().view();
    const auto bFluxV = faceFlux.boundaryData().value().view();
    auto values = ls.matrix().values().view();
    const auto nBoundaryFaces = mesh.nBoundaryFaces();

    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            const auto ownRow = ownV[bfi];
            const auto phi_f = bFluxV[bfi];
            Kokkos::atomic_sub(
                &values[ma.diagIdx(ownRow)], phi_f * scaling[ownRow] * one<AssemblyType>()
            );
        },
        "BoundedDiv::applyBoundedDiagBoundary"
    );
}

// diag[i] -= (isOwner ? +φ_f : -φ_f) * scaling[i] over proc boundary faces.
// On a proc boundary the local cell can be either the owner or the neighbor
// of the cross-rank face; the `weights` boundary stream encodes which via
// isOwner[bcfacei] > 0 (matching the existing computeDivProcBoundImpl).
template<typename FieldValueType, typename AssemblyType>
void applyBoundedDiagProcBoundary(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const UnstructuredMesh& mesh,
    const dsl::Coeff scaling
)
{
    const auto nProcBoundaryFaces = mesh.nProcBoundaryFaces();
    if (nProcBoundaryFaces == 0) return;

    const auto exec = ls.exec();
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    const auto ownV = mesh.boundaryMesh().faceOwners().view();
    const auto isOwnerV = mesh.boundaryMesh().weights().view();
    const auto bFluxV = faceFlux.boundaryData().value().view();
    auto values = ls.matrix().values().view();

    parallelFor(
        exec,
        {0, nProcBoundaryFaces},
        NEON_LAMBDA(const localIdx procFacei) {
            const auto bcfacei = nBoundaryFaces + procFacei;
            const auto cell = ownV[bcfacei];
            const auto isOwnerFace = isOwnerV[bcfacei] > 0.0;
            const auto phi_f = bFluxV[bcfacei];
            const auto outflow = isOwnerFace ? phi_f : -phi_f;
            Kokkos::atomic_sub(
                &values[ma.diagIdx(cell)], outflow * scaling[cell] * one<AssemblyType>()
            );
        },
        "BoundedDiv::applyBoundedDiagProcBoundary"
    );
}

// result[i] -= surfaceIntegrate(F)[i] * V[i] * ψ[i] * scaling[i]
//            = (Σ_f φ_f over cell i) * ψ[i] * scaling[i]
// for the explicit form. Same face-by-face accumulation as the implicit
// helpers above — only the per-cell multiplier changes (ψ instead of the
// matrix diagonal slot).
template<typename FieldValueType>
void applyBoundedExplicit(
    Vector<FieldValueType>& result,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const UnstructuredMesh& mesh,
    const dsl::Coeff scaling
)
{
    const auto exec = phi.exec();
    const auto resultV = result.view();
    const auto phiInternalV = phi.internalVector().view();
    const auto cellVolumesV = mesh.cellVolumes().view();

    // First, accumulate Σ_f φ_f per cell into a scratch vector. Doing the
    // explicit form by direct face-scatter onto result[] would risk reading
    // phi[i] in the middle of a partial sum; staging the divergence per cell
    // first keeps the (Σ φ) × ψ scaling correct under any face order.
    Vector<scalar> sumPhi(exec, result.size(), scalar(0));
    auto sumPhiV = sumPhi.view();

    {
        const auto [fluxV, ownV, neiV] =
            views(faceFlux.internalVector(), mesh.faceOwners(), mesh.faceNeighbors());
        parallelFor(
            exec,
            {0, mesh.nInternalFaces()},
            NEON_LAMBDA(const localIdx facei) {
                const auto phi_f = fluxV[facei];
                Kokkos::atomic_add(&sumPhiV[ownV[facei]], phi_f);
                Kokkos::atomic_sub(&sumPhiV[neiV[facei]], phi_f);
            },
            "BoundedDiv::accumulateSumPhiInternal"
        );
    }

    {
        const auto ownV = mesh.boundaryMesh().faceOwners().view();
        const auto bFluxV = faceFlux.boundaryData().value().view();
        parallelFor(
            exec,
            {0, mesh.nBoundaryFaces()},
            NEON_LAMBDA(const localIdx bfi) {
                Kokkos::atomic_add(&sumPhiV[ownV[bfi]], bFluxV[bfi]);
            },
            "BoundedDiv::accumulateSumPhiBoundary"
        );
    }

    const auto nProc = mesh.nProcBoundaryFaces();
    if (nProc > 0)
    {
        const auto nBnd = mesh.nBoundaryFaces();
        const auto ownV = mesh.boundaryMesh().faceOwners().view();
        const auto isOwnerV = mesh.boundaryMesh().weights().view();
        const auto bFluxV = faceFlux.boundaryData().value().view();
        parallelFor(
            exec,
            {0, nProc},
            NEON_LAMBDA(const localIdx procFacei) {
                const auto bcfacei = nBnd + procFacei;
                const auto cell = ownV[bcfacei];
                const auto isOwnerFace = isOwnerV[bcfacei] > 0.0;
                const auto phi_f = bFluxV[bcfacei];
                Kokkos::atomic_add(&sumPhiV[cell], isOwnerFace ? phi_f : -phi_f);
            },
            "BoundedDiv::accumulateSumPhiProc"
        );
    }

    // result[i] -= sumPhi[i] / V[i] * V[i] * ψ[i] * scaling[i]
    //           = sumPhi[i] * ψ[i] * scaling[i]
    parallelFor(
        exec,
        {0, result.size()},
        NEON_LAMBDA(const localIdx celli) {
            // suppress "unused" on cellVolumes — kept around in case future
            // changes need an explicit V-divided coefficient.
            (void)cellVolumesV;
            resultV[celli] -= sumPhiV[celli] * phiInternalV[celli] * scaling[celli];
        },
        "BoundedDiv::subtractBoundedCorrection"
    );
}

} // namespace


template<typename FieldValueType, typename AssemblyType>
BoundedDiv<FieldValueType, AssemblyType>::BoundedDiv(
    const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs
)
    : Base(exec, mesh),
      inner_(DivOperatorFactory<FieldValueType, AssemblyType>::create(exec, mesh, inputs))
{
    // The leading "bounded" token was consumed by the outer factory's
    // create(); inner_ then consumes "Gauss" and its sub-scheme handles
    // the remaining tokens (e.g. "upwind").
}

template<typename FieldValueType, typename AssemblyType>
BoundedDiv<FieldValueType, AssemblyType>::BoundedDiv(
    const Executor& exec,
    const UnstructuredMesh& mesh,
    std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>> inner
)
    : Base(exec, mesh), inner_(std::move(inner))
{}

template<typename FieldValueType, typename AssemblyType>
void BoundedDiv<FieldValueType, AssemblyType>::div(
    VolumeField<FieldValueType>& divPhi,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff operatorScaling
) const
{
    inner_->div(divPhi, faceFlux, phi, operatorScaling);
    applyBoundedExplicit<FieldValueType>(
        divPhi.internalVector(), faceFlux, phi, this->mesh_, operatorScaling
    );
}

template<typename FieldValueType, typename AssemblyType>
void BoundedDiv<FieldValueType, AssemblyType>::div(
    Vector<FieldValueType>& divPhi,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff operatorScaling
) const
{
    inner_->div(divPhi, faceFlux, phi, operatorScaling);
    applyBoundedExplicit<FieldValueType>(divPhi, faceFlux, phi, this->mesh_, operatorScaling);
}

template<typename FieldValueType, typename AssemblyType>
void BoundedDiv<FieldValueType, AssemblyType>::div(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff operatorScaling
) const
{
    inner_->div(ls, faceFlux, phi, operatorScaling);
    applyBoundedDivDiagonal<FieldValueType, AssemblyType>(
        ls, faceFlux, this->mesh_, operatorScaling
    );
}

template<typename FieldValueType, typename AssemblyType>
VolumeField<FieldValueType> BoundedDiv<FieldValueType, AssemblyType>::div(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff operatorScaling
) const
{
    auto divPhi = inner_->div(faceFlux, phi, operatorScaling);
    applyBoundedExplicit<FieldValueType>(
        divPhi.internalVector(), faceFlux, phi, this->mesh_, operatorScaling
    );
    return divPhi;
}

template<typename FieldValueType, typename AssemblyType>
std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>>
BoundedDiv<FieldValueType, AssemblyType>::clone() const
{
    // Custom clone — BoundedDiv owns a unique_ptr<inner> which can't be
    // default-copied, so we deep-copy via the inner factory's clone() and
    // hand the result to the private inner-bypassing constructor.
    return std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>>(
        new BoundedDiv(this->exec_, this->mesh_, inner_->clone())
    );
}

template<typename FieldValueType, typename AssemblyType>
void applyBoundedDivDiagonal(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const UnstructuredMesh& mesh,
    const dsl::Coeff scaling
)
{
    applyBoundedDiagInternal<FieldValueType, AssemblyType>(ls, faceFlux, mesh, scaling);
    applyBoundedDiagBoundary<FieldValueType, AssemblyType>(ls, faceFlux, mesh, scaling);
    applyBoundedDiagProcBoundary<FieldValueType, AssemblyType>(ls, faceFlux, mesh, scaling);
}

template class BoundedDiv<scalar>;
template class BoundedDiv<Vec3>;
template class BoundedDiv<Vec3, scalar>;

template void applyBoundedDivDiagonal<scalar, scalar>(
    la::LinearSystem<scalar, scalar>&,
    const SurfaceField<scalar>&,
    const UnstructuredMesh&,
    const dsl::Coeff
);
template void applyBoundedDivDiagonal<Vec3, Vec3>(
    la::LinearSystem<Vec3, Vec3>&,
    const SurfaceField<scalar>&,
    const UnstructuredMesh&,
    const dsl::Coeff
);
template void applyBoundedDivDiagonal<Vec3, scalar>(
    la::LinearSystem<scalar, Vec3>&,
    const SurfaceField<scalar>&,
    const UnstructuredMesh&,
    const dsl::Coeff
);

} // namespace NeoN::finiteVolume::cellCentred
