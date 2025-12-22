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
    const fvcc::SurfaceField<scalar>& phiCorr, NeoN::la::LinearSystem<scalar, NeoN::localIdx>& ls
)
{
    const auto& mesh = phiCorr.mesh();
    const auto exec = phiCorr.exec();

    auto [rhs, owner, neighbour, phiV] =
        views(ls.rhs(), mesh.faceOwner(), mesh.faceNeighbour(), phiCorr.internalVector());

    const auto nInternalFaces = mesh.nInternalFaces();

    NeoN::parallelFor(
        exec,
        {NeoN::localIdx(0), NeoN::localIdx(nInternalFaces)},
        KOKKOS_LAMBDA(const NeoN::localIdx f) {
            const auto flux = phiV[f];
            rhs[owner[f]] -= flux;
            rhs[neighbour[f]] += flux;
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
     * @param U      Velocity field
     * @param phi    Face flux field
     * @param dt     Time step size
     */
    DdtPhiCorr(
        const DdtScheme& scheme, const VolVectorField& U, const SurfScalarField& phi, scalar dt
    )
        : scheme_(scheme), U_(U), phi_(phi), dt_(dt)
    {}

    void operator()(const NeoN::la::SparsityPattern&, LinearSystem& ls) override
    {
        auto phiCorr = scheme_.ddtPhiCorr(U_, phi_, dt_);
        detail::applyDivToRhs(phiCorr, ls);
    }

private:

    const DdtScheme& scheme_;
    const VolVectorField& U_;
    const SurfScalarField& phi_;
    scalar dt_;
};

} // namespace NeoN::dsl
