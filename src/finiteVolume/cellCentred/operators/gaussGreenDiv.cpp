// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"
#include <julia.h>

namespace NeoN::finiteVolume::cellCentred
{

/* @brief free standing function implementation of the divergence operator
** ie computes 1/V \sum_f S_f \cdot \phi_f
** where S_f is the face normal flux of a given face
**  phi_f is the face interpolate value
**
**
** @param exec The executor
** @param nInternalFaces - number of internal faces
** @param nBoundaryFaces - number of boundary faces
** @param neighbors - mapping from face id to neighbors cell id
** @param owner - mapping from face id to owner cell id
** @param faceOwners - mapping from boundary face id to owner cell id
** @param faceFlux - flux on cell faces
** @param phiF - flux on cell faces
** @param v - cell volumes
** @param res - view holding the result
** @param operatorScaling - any additional coefficients
*/
template<typename ValueType>
void computeDiv(
    const Executor& exec,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    View<const localIdx> neighbors,
    View<const localIdx> owners,
    View<const localIdx> faceOwners,
    View<const scalar> faceFlux,
    View<const scalar> bFaceFlux,
    View<const ValueType> phiF,
    View<const ValueType> bPhiF,
    View<const scalar> v,
    View<ValueType> res,
    const dsl::Coeff operatorScaling
)
{
    auto nCells = v.size();

    // Green-Gauss divergence theorem: ∇·(F φ)_C = (1/V_C) * sum_f F_f * φ_f
    //
    // F_f = faceFlux[f] is the signed scalar flux through face f.
    // S_f points from owner to neighbor by construction, so F_f = U · S_f:
    //   F_f > 0 → flux leaving the owner cell and entering the neighbor cell.
    //
    // The DIVERGENCE at a cell measures net outward flux, so:
    //   owner cell:     F_f is outward (S_f points away from owner) → +F_f * φ_f  (add)
    //   neighbor cell: F_f is inward  (S_f points into neighbor)  → −F_f * φ_f  (subtract)
    //
    // This computes +∇·(F φ) (positive divergence form).
    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx i) {
            ValueType flux = faceFlux[i] * phiF[i];
            Kokkos::atomic_add(&res[owners[i]], flux);    // F_f outward from owner
            Kokkos::atomic_sub(&res[neighbors[i]], flux); // F_f inward to neighbor
        },
        "sumFluxesInternal"
    );

    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto own = faceOwners[bfi];
            ValueType valueOwn = bFaceFlux[bfi] * bPhiF[bfi];
            Kokkos::atomic_add(&res[own], valueOwn); // boundary face: F_f outward from owner
        },
        "sumFluxesBoundary"
    );

    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) { res[celli] *= operatorScaling[celli] / v[celli]; },
        "normalizeFluxes"
    );
}

template<typename ValueType>
void computeDivJulia(
    const Executor& exec,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    View<const localIdx> neighbour,
    View<const localIdx> owner,
    View<const localIdx> faceCells,
    View<const scalar> faceFlux,
    View<const ValueType> phiF,
    View<const scalar> v,
    View<ValueType> res,
    const dsl::Coeff operatorScaling
)
{
    auto nCells = v.size();
    jl_eval_string(R"(
        function get_operator(tokens::Vector{String})
            if tokens[1] == "DivOperator"
            else # only Laplacian for now
                return Laplace{Float64}(1.0)
            end
            operator = ifelse(tokens[1] == "DivOperator", CentralDiffScheme{Float64},UpwindScheme{Float64}) 
            divscheme = ifelse(tokens[2] == "linear", CentralDiffScheme{Float64},UpwindScheme{Float64}) 
            div = Div{Float64,divscheme}(divscheme(), 1.0) 
            return String(Symbol(div))
        end
        function SOAFusedFaceBasedAssembly(numInteriorFaces::Int32, owner::Ptr{Cvoid}, neighbor::Ptr{Cvoid},  ) where {P<:AbstractFloat}
        #function SOAFusedFaceBasedAssembly(input::SOAMatrixAssemblyInput{P}, vals::Vector{P}, RHS::Vector{P}, fused_pde::DiffEq) where {P<:AbstractFloat}
            nu = input.nu
            faces = input.faces
            U_b = input.U_boundary
            U = input.U_internal
            nCells = length(input.cells.index)
            @inbounds for iFace in 1:input.numInteriorFaces
                @inbounds iOwner = faces.iOwner[iFace]
                @inbounds iNeighbor = faces.iNeighbor[iFace]
                @inbounds valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor], faces.Sf[iFace], nu[iOwner], faces.gDiff[iFace], zero(P), zero(P))

                @inbounds vals[faces.ownerIdx[iFace]] += valueUpper
                @inbounds vals[faces.neighborIdx[iFace]] += valueLower
                @inbounds vals[faces.neighborRelNeighborIdx[iFace]] += valueUpper
                @inbounds vals[faces.ownerRelOwnerIdx[iFace]] += valueLower
            end
            @inbounds for bFace in numInteriorFaces:length(owner)
            @inbounds for iBoundary in eachindex(input.boundaries)
                if U_b[iBoundary].type != "fixedValue"
                    continue
                end
                @inbounds theBoundary = input.boundaries[iBoundary]
                startFace = theBoundary.startFace + 1
                endFace = startFace + theBoundary.nFaces
                for iFace in startFace:endFace-1
                    @inbounds relativeFaceIndex = iFace - input.boundaries[iBoundary].startFace
                    diag, rhsx, rhsy, rhsz = fused_pde(U_b[iBoundary].values[relativeFaceIndex], faces.Sf[iFace], nu[faces.iOwner[iFace]], faces.gDiff[iFace], zero(P), zero(P), zero(P), zero(P))

                    @inbounds vals[faces.ownerIdx[iFace]] += diag
                    # RHS/Source
                    @inbounds RHS[faces.iOwner[iFace]] += rhsx
                    @inbounds RHS[faces.iOwner[iFace]+nCells] += rhsy
                    @inbounds RHS[faces.iOwner[iFace]+nCells+nCells] += rhsz
                end
            end
            return vals, RHS
        end # function batchedFaceBasedAssembly
    )");
    if (std::holds_alternative<CPUExecutor>(exec) || std::holds_alternative<SerialExecutor>(exec))
    {
        for (localIdx i = 0; i < nInternalFaces; i++)
        {
            ValueType flux = faceFlux[i] * phiF[i];
            res[owner[i]] += flux;
            res[neighbour[i]] -= flux;
        }

        for (localIdx i = nInternalFaces; i < nInternalFaces + nBoundaryFaces; i++)
        {
            auto own = faceCells[i - nInternalFaces];
            ValueType valueOwn = faceFlux[i] * phiF[i];
            res[own] += valueOwn;
        }

        // TODO does it make sense to store invVol and multiply?
        for (localIdx celli = 0; celli < nCells; celli++)
        {
            res[celli] *= operatorScaling[celli] / v[celli];
        }
    }
}

template<typename ValueType>
void computeDivExp(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation<ValueType>& surfInterp,
    Vector<ValueType>& divPhi,
    const dsl::Coeff operatorScaling
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto exec = phi.exec();
    SurfaceField<ValueType> phif(
        exec, "phif", mesh, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh)
    );
    // TODO: remove or implement
    // fill(phif.internalVector(), NeoN::zero<ValueType>::value);
    surfInterp.interpolate(faceFlux, phi, phif);

    // TODO: currently we just copy the boundary values over
    phif.boundaryData().value() = phi.boundaryData().value();

    auto nInternalFaces = mesh.nInternalFaces();
    auto nBoundaryFaces = mesh.nBoundaryFaces();
    computeDiv<ValueType>(
        exec,
        nInternalFaces,
        nBoundaryFaces,
        mesh.faceNeighbors().view(),
        mesh.faceOwners().view(),
        mesh.boundaryMesh().faceOwners().view(),
        faceFlux.internalVector().view(),
        faceFlux.boundaryData().value().view(),
        phif.internalVector().view(),
        phif.boundaryData().value().view(),
        mesh.cellVolumes().view(),
        divPhi.view(),
        operatorScaling
    );
}

#define NF_DECLARE_COMPUTE_EXP_DIV(TYPENAME)                                                       \
    template void computeDivExp<TYPENAME>(                                                         \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        Vector<TYPENAME>&,                                                                         \
        const dsl::Coeff                                                                           \
    )

NF_DECLARE_COMPUTE_EXP_DIV(scalar);
NF_DECLARE_COMPUTE_EXP_DIV(Vec3);

template<typename ValueType>
void computeDivBoundImp(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceField<scalar>& weights,
    const dsl::Coeff operatorScaling
)
{
    const auto exec = phi.exec();
    const auto& mesh = phi.mesh();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const auto [ownV, deltaCoeffs] =
        views(mesh.boundaryMesh().faceOwners(), mesh.boundaryMesh().deltaCoeffs());

    auto values = ls.matrix().values().view();

    auto [bFaceFluxV, bweights, refGradient, valueFraction, refValue] = views(
        faceFlux.boundaryData().value(),
        weights.boundaryData().value(),
        phi.boundaryData().refGrad(),
        phi.boundaryData().valueFraction(),
        phi.boundaryData().refValue()
    );

    auto rhs = ls.rhs().view();
    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.boundaryMatrix().values().view();

    const auto nBoundaryFaces = mesh.nBoundaryFaces();
    parallelFor(
        exec,
        {0, nBoundaryFaces},
        NEON_LAMBDA(const localIdx bfi) {
            auto ownRow = ownV[bfi];

            auto ownCoeff = operatorScaling[ownRow];

            auto refValFrac = valueFraction[bfi];
            auto refGradFrac = 1.0 - refValFrac;

            auto flux =
                bFaceFluxV[bfi] * -bweights[bfi] * ownCoeff * refGradFrac * one<ValueType>();

            // since upper triangular value is "outside" of system matrix
            // it is stored separately in bMatrix
            bValues[bfi] += flux;
            // diagonal contribution
            Kokkos::atomic_sub(&values[ma.diagIdx(ownRow)], flux);

            // Explicit RHS contribution from the mixed BC:
            //   φ_f = refValFrac * refValue               (Dirichlet part)
            //       + refGradFrac * (φ_C + refGradient/δ)  (Neumann part)
            // The implicit valFrac2 * φ_C term is handled via fluxContrib above.
            // bweights converts the Dirichlet face value to a cell-to-face flux contribution;
            // the Neumann gradient correction (refGradient/δ) enters directly as a known increment.
            auto valueRhs =
                (bweights[bfi] * bFaceFluxV[bfi] * ownCoeff * (refValFrac * refValue[bfi]))
                + refGradFrac * refGradient[bfi] * (1 / deltaCoeffs[bfi]);
            Kokkos::atomic_sub(&rhs[ownRow], valueRhs);
            bRhs[bfi] += valueRhs;
        },
        "computeInterfaceGaussGreenDivCoefficients"
    );
}


template<typename ValueType>
void computeDivIntImp(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceField<scalar>& weights,
    const dsl::Coeff coeff
)
{
    const UnstructuredMesh& mesh = phi.mesh();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nCells = mesh.nCells();
    const auto exec = phi.exec();

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const auto [fluxV, weightsV, ownV, neiV, surfFaceCells] = views(
        faceFlux.internalVector(),
        weights.internalVector(),
        mesh.faceOwners(),
        mesh.faceNeighbors(),
        mesh.boundaryMesh().faceOwners()
    );
    auto values = ls.matrix().values().view();

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            // row and column indices
            auto ownRow = ownV[facei];
            auto neiRow = neiV[facei];

            // operator sign coefficient  handles: = +/- div
            auto ownCoeff = coeff[ownRow];
            auto neiCoeff = coeff[neiRow];

            // Conservative Gauss-Green divergence assembly.
            // S_f points from owner to neighbor by construction, so F_f < 0 means
            // flux leaves the owner cell and enters the neighbor cell.
            //
            // Decompose face flux via linear interpolation:
            //   ownFluxContrib = w * F_f     — part attributed to the owner cell value
            //   neiFluxContrib = (1-w) * F_f — part attributed to the neighbor cell value
            auto ownFluxContrib = -fluxV[facei] * weightsV[facei] * one<ValueType>();
            auto neiFluxContrib = +fluxV[facei] * (1.0 - weightsV[facei]) * one<ValueType>();

            // triangular coefficients - neighbor -> lower, owner -> upper
            values[ma.lowerIdx(neiRow, facei)] += ownFluxContrib * neiCoeff;
            values[ma.upperIdx(ownRow, facei)] += neiFluxContrib * ownCoeff;

            // diagonal contribution is negative sum of offdiagonal coefficients
            Kokkos::atomic_sub(&values[ma.diagIdx(ownRow)], ownFluxContrib * ownCoeff);
            Kokkos::atomic_sub(&values[ma.diagIdx(neiRow)], neiFluxContrib * neiCoeff);
        },
        "computeLocalGaussGreenDivCoefficients"
    );
};

#define NN_DECLARE_COMPUTE_IMP_DIV(TYPENAME)                                                       \
    template void computeDivIntImp<TYPENAME>(                                                      \
        la::LinearSystem<TYPENAME>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const dsl::Coeff                                                                           \
    );                                                                                             \
    template void computeDivBoundImp<TYPENAME>(                                                    \
        la::LinearSystem<TYPENAME>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceField<scalar>&,                                                               \
        const dsl::Coeff                                                                           \
    )

NN_DECLARE_COMPUTE_IMP_DIV(scalar);
NN_DECLARE_COMPUTE_IMP_DIV(Vec3);

template class GaussGreenDiv<scalar>;
template class GaussGreenDiv<Vec3>;

};
