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
** @param neighbour - mapping from face id to neighbour cell id
** @param owner - mapping from face id to owner cell id
** @param faceCells - mapping from boundary face id to owner cell id
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
    // check if the executor is GPU
    if (std::holds_alternative<SerialExecutor>(exec))
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
    else
    {
        parallelFor(
            exec,
            {0, nInternalFaces},
            NEON_LAMBDA(const localIdx i) {
                ValueType flux = faceFlux[i] * phiF[i];
                Kokkos::atomic_add(&res[owner[i]], flux);
                Kokkos::atomic_sub(&res[neighbour[i]], flux);
            },
            "sumFluxesInternal"
        );

        parallelFor(
            exec,
            {nInternalFaces, nInternalFaces + nBoundaryFaces},
            NEON_LAMBDA(const localIdx i) {
                auto own = faceCells[i - nInternalFaces];
                ValueType valueOwn = faceFlux[i] * phiF[i];
                Kokkos::atomic_add(&res[own], valueOwn);
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
        mesh.faceNeighbour().view(),
        mesh.faceOwner().view(),
        mesh.boundaryMesh().faceCells().view(),
        faceFlux.internalVector().view(),
        phif.internalVector().view(),
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
void computeDivImp(
    la::LinearSystem<ValueType>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation<ValueType>& surfInterp,
    const dsl::Coeff operatorScaling
)
{
    std::cout << "computeDivImp\n";
    const UnstructuredMesh& mesh = phi.mesh();
    const auto matIt = ls.faceToMatrixAddress();
    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nCells = mesh.nCells();
    const auto exec = phi.exec();
    const auto weights = surfInterp.weight(faceFlux, phi);
    const auto
        [faceFluxV,
         weightsV,
         owner,
         neighbour,
         surfFaceCells,
         diagOffs,
         ownOffs,
         neiOffs,
         rowOffs] =
            views(
                faceFlux.internalVector(),
                weights.internalVector(),
                mesh.faceOwner(),
                mesh.faceNeighbour(),
                mesh.boundaryMesh().faceCells(),
                matIt->diagOffset(),
                matIt->ownerOffset(),
                matIt->neighbourOffset(),
                matIt->sparsityPattern()->rowOffs()
            );
    const auto
        [JUfaceFluxV,
         JUweightsV,
         JUowner,
         JUneighbour,
         JUsurfFaceCells,
         JUdiagOffs,
         JUownOffs,
         JUneiOffs,
         JUrowOffs] =
            juliaPtrs(
                faceFlux.internalVector(),
                weights.internalVector(),
                mesh.faceOwner(),
                mesh.faceNeighbour(),
                mesh.boundaryMesh().faceCells(),
                matIt->diagOffset(),
                matIt->ownerOffset(),
                matIt->neighbourOffset(),
                matIt->sparsityPattern()->rowOffs()
            );
    auto rhs = ls.rhs().view();
    auto values = ls.matrix().values().view();

    // auto jval = Vector<double>(ls.exec(), nInternalFaces * 2 + nCells);
    // auto jvv = jval.view();
    // auto JUvalues = jval.juliaPtr();
    // jl_eval_string(R"(
    //     function computeLocalGaussGreenDivCoefficients(
    //         numInteriorFaces::Int32,
    //         numCells::Int32,
    //         faceFluxV::Vector{Float64},
    //         weightsV::Vector{Float64},
    //         owner::Vector{Int32},
    //         neighbour::Vector{Int32},
    //         diagOffs::Vector{UInt8},
    //         ownOffs::Vector{UInt8},
    //         neiOffs::Vector{UInt8},
    //         rowOffs::Vector{Int32},
    //         vals::Vector{Float64}
    //     )
    //         for facei in 1:numInteriorFaces
    //             @inbounds own = owner[facei]
    //             @inbounds nei = neighbour[facei]
    //             # @inbounds valueUpper, valueLower = fused_pde(U[iOwner], U[iNeighbor],
    //             faces.Sf[iFace], nu, 0.0, 0.0, 0.0) valueUpper = faceFluxV[facei] *
    //             -weightsV[facei] valueLower = faceFluxV[facei] * (1 - weightsV[facei])

    //             @inbounds rowNeiStart = rowOffs[nei]
    //             @inbounds rowOwnStart = rowOffs[own]

    //             @inbounds vals[rowOwnStart+diagOffs[own]] -= valueUpper
    //             @inbounds vals[rowNeiStart+diagOffs[nei]] -= valueLower
    //             @inbounds vals[rowNeiStart+neiOffs[facei]] += valueUpper
    //             @inbounds vals[rowOwnStart+ownOffs[facei]] += valueLower
    //         end
    //     end
    // )");
    // jl_function_t* func = jl_get_function(jl_main_module,
    // "computeLocalGaussGreenDivCoefficients"); size_t s = 11; jl_value_t* args[s]; args[0] =
    // jl_box_int32(nInternalFaces); args[1] = jl_box_int32(nCells); args[2] =
    // (jl_value_t*)JUfaceFluxV; args[3] = (jl_value_t*)JUweightsV; args[4] = (jl_value_t*)JUowner;
    // args[5] = (jl_value_t*)JUneighbour;
    // args[6] = (jl_value_t*)JUdiagOffs;
    // args[7] = (jl_value_t*)JUownOffs;
    // args[8] = (jl_value_t*)JUneiOffs;
    // args[9] = (jl_value_t*)JUrowOffs;
    // args[10] = (jl_value_t*)JUvalues;
    // jl_call(func, args, s);
    // jl_value_t* exc = jl_exception_occurred();
    // if (exc)
    // {
    //     std::cerr << ": Julia exception: " << jl_typeof_str(exc) << std::endl;
    // }

    parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            auto own = owner[facei];
            auto nei = neighbour[facei];

            auto operatorScalingNei = operatorScaling[nei];
            auto operatorScalingOwn = operatorScaling[own];
            auto rowNeiStart = rowOffs[nei];
            auto rowOwnStart = rowOffs[own];

            auto valueUpper = faceFluxV[facei] * -weightsV[facei] * one<ValueType>();
            // matrix.values[matIt.upperIdx(nei, facei)] += valueUpper * operatorScalingNei;


            values[rowNeiStart + neiOffs[facei]] += valueUpper; // * operatorScalingNei;
            Kokkos::atomic_sub(
                &values[rowOwnStart + diagOffs[own]], valueUpper // * operatorScalingOwn
            );

            // add owner contribution lower
            auto valueLower = faceFluxV[facei] * (1 - weightsV[facei]) * one<ValueType>();
            // matrix.values[matIt.lowerIdx(own, facei)] += valueLower * operatorScalingOwn;


            values[rowOwnStart + ownOffs[facei]] += valueLower; // * operatorScalingOwn;

            Kokkos::atomic_sub(
                &values[rowNeiStart + diagOffs[nei]], valueLower // * operatorScalingNei
            );


            // if (rowOwnStart == 4 || rowNeiStart == 4)
            // {
            //     std::cout << "face 1: " << rowNeiStart << ", " << rowOwnStart << std::endl;
            //     std::cout << "face 1: " << unsigned(diagOffs[nei]) << ", "
            //               << unsigned(diagOffs[own]) << std::endl;
            //     std::cout << "face 1: " << valueUpper << ", " << valueLower << std::endl;
            //     std::cout << "face 1: " << faceFluxV[facei] << ", " << weightsV[facei] <<
            //     std::endl;
            // }
        },
        "computeLocalGaussGreenDivCoefficients"
    );
    return;
    std::cout << "values[0]: " << values[0] << std::endl;
    std::cout << "values[1]: " << values[1] << std::endl;
    std::cout << "values[2]: " << values[2] << std::endl;
    std::cout << "values[3]: " << values[3] << std::endl;
    std::cout << "values[4]: " << values[4] << std::endl;
    auto [bweights, refGradient, value, valueFraction, refValue, deltaCoeffs] = views(
        weights.boundaryData().value(),
        phi.boundaryData().refGrad(),
        phi.boundaryData().value(),
        phi.boundaryData().valueFraction(),
        phi.boundaryData().refValue(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    auto bRhs = ls.boundaryRhs().view();
    auto bValues = ls.boundaryMatrix().values().view();

    parallelFor(
        exec,
        {nInternalFaces, faceFluxV.size()},
        NEON_LAMBDA(const localIdx facei) {
            auto bcfacei = facei - nInternalFaces;
            auto flux = bweights[bcfacei] * faceFluxV[facei];

            auto own = surfFaceCells[bcfacei];
            auto rowOwnStart = rowOffs[own];
            auto operatorScalingOwn = operatorScaling[own];

            auto valFrac1 = valueFraction[bcfacei];
            auto valFrac2 = 1.0 - valFrac1;

            auto valueMat = flux * operatorScalingOwn * valFrac2 * one<ValueType>();

            Kokkos::atomic_add(&values[rowOwnStart + diagOffs[own]], valueMat);
            bValues[bcfacei] = valueMat;

            auto valueRhs = (flux * operatorScalingOwn * (valFrac1 * refValue[bcfacei]))
                          + valFrac2 * refGradient[bcfacei] * (1 / deltaCoeffs[bcfacei]);

            Kokkos::atomic_sub(&rhs[own], valueRhs);
            bRhs[bcfacei] = valueRhs;
        },
        "computeInterfaceGaussGreenDivCoefficients"
    );
};

#define NN_DECLARE_COMPUTE_IMP_DIV(TYPENAME)                                                       \
    template void computeDivImp<TYPENAME>(                                                         \
        la::LinearSystem<TYPENAME>&,                                                               \
        const SurfaceField<scalar>&,                                                               \
        const VolumeField<TYPENAME>&,                                                              \
        const SurfaceInterpolation<TYPENAME>&,                                                     \
        const dsl::Coeff                                                                           \
    )

NN_DECLARE_COMPUTE_IMP_DIV(scalar);
NN_DECLARE_COMPUTE_IMP_DIV(Vec3);

};
