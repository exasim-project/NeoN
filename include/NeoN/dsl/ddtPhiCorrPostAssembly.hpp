// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/dsl/expression.hpp"

#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoN::dsl
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace detail
{

inline void applyDivToRhs(
    const fvcc::SurfaceField<scalar>& fluxCorr, NeoN::la::LinearSystem<scalar, NeoN::localIdx>& ls
)
{
    const auto& mesh = fluxCorr.mesh();
    const auto exec = fluxCorr.exec();

    auto [rhsV, ownerV, neighbourV, fluxCorrV] =
        views(ls.rhs(), mesh.faceOwner(), mesh.faceNeighbour(), fluxCorr.internalVector());

    const auto nInternalFaces = mesh.nInternalFaces();

    NeoN::parallelFor(
        exec,
        {NeoN::localIdx(0), NeoN::localIdx(nInternalFaces)},
        KOKKOS_LAMBDA(const NeoN::localIdx f) {
            const auto flux = fluxCorrV[f];
            rhsV[ownerV[f]] -= fluxCorrV[f];
            rhsV[neighbourV[f]] += fluxCorrV[f];
        },
        "postAssembly::DdtPhiCorr"
    );
}

} // namespace detail

template<typename ValueType>
class DdtPhiCorr final : public NeoN::dsl::PostAssemblyBase<ValueType>
{
public:

    static_assert(
        std::is_same_v<ValueType, NeoN::scalar>,
        "DdtPhiCorr postAssembly is only valid for scalar equations"
    );

    using VolVectorField = fvcc::VolumeField<Vec3>;
    using SurfScalarField = fvcc::SurfaceField<scalar>;
    using DdtScheme = NeoN::timeIntegration::DdtScheme;
    using LinearSystem = NeoN::la::LinearSystem<ValueType, NeoN::localIdx>;

    /**
     * @param scheme ddtScheme used for discretisation
     * @param u      Velocity field
     * @param flux    Face flux field
     * @param dt     Time step size
     */
    DdtPhiCorr(
        const DdtScheme& scheme, const VolVectorField& u, const SurfScalarField& flux, scalar dt
    )
        : scheme_(scheme), U_(u), flux_(flux), dt_(dt)
    {}

    void operator()(const NeoN::la::SparsityPattern&, LinearSystem& ls) override
    {
        auto fluxCorr = scheme_.ddtFluxCorr(U_, flux_, dt_);
        detail::applyDivToRhs(fluxCorr, ls);
    }

private:

    const DdtScheme& scheme_;
    const VolVectorField& U_;
    const SurfScalarField& flux_;
    scalar dt_;
};

} // namespace NeoN::dsl
