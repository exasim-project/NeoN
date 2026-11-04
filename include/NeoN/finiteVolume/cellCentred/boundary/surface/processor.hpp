// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary
{

namespace detail
{
// One-time (set): zero the unused mixed-BC coefficients (refGrad / valueFraction / refValue) on
// the processor patch. Unlike the volume BC there is no per-iteration owner-cell -> face seed
// (see processor.cpp), so this is all the surface proc BC does besides the halo exchange.
template<typename ValueType>
void setProcBoundaryCoefficients(
    Field<ValueType>& domainVector, std::pair<localIdx, localIdx> range
);

extern template void
setProcBoundaryCoefficients<scalar>(Field<scalar>&, std::pair<localIdx, localIdx>);
extern template void setProcBoundaryCoefficients<Vec3>(Field<Vec3>&, std::pair<localIdx, localIdx>);
}

template<typename ValueType>
class Processor : public SurfaceBoundaryFactory<ValueType>::template Register<Processor<ValueType>>
{
    using Base = SurfaceBoundaryFactory<ValueType>::template Register<Processor<ValueType>>;

public:

    Processor(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID), mesh_(mesh)
    {}

    // One-time initialisation: the unused mixed-BC coefficients never change after construction.
    virtual void set(Field<ValueType>& domainVector) final
    {
        detail::setProcBoundaryCoefficients(domainVector, this->range());
    }

    // Per iteration: NO-OP. The proc-tail already holds the LOCAL face value (written by the
    // operator that produced the field — flux() / updateFaceVelocity() / interpolation — or by
    // constructFrom() at construction), and that LOCAL value is exactly what every consumer needs
    // (computeDivProcBoundImpl, surfaceIntegrate, upwind weights all use the owner-rank outward
    // flux). Crucially, a face flux is ANTISYMMETRIC across a processor interface — the neighbour
    // rank's value for the shared face is -F (opposite Sf orientation). A symmetric halo exchange
    // (BoundaryData::communicate does send AND receive-overwrite, as needed for cell/volume data)
    // would therefore OVERWRITE the correct local +F with the neighbour's -F, sign-flipping the
    // proc-face flux. That injects a spurious local divergence at every processor-boundary cell
    // (cancels globally so conservation holds, but inflates the per-cell divergence and the
    // distributed momentum/pressure residual). Surface processor patches must keep their local
    // value, so this update is intentionally empty.
    virtual void update([[maybe_unused]] Field<ValueType>& domainVector) final {}

    // Full correction = one-time set() + per-iteration update(). Kept for any direct caller; the
    // field's correctBoundaryConditions() calls set()/update() separately.
    virtual void correctBoundaryCondition([[maybe_unused]] Field<ValueType>& domainVector) final
    {
        set(domainVector);
        update(domainVector);
    }


    static std::string name() { return "processor"; }

    static std::string doc()
    {
        return "Set processor boundary values from the corresponding processor-neighbour data.";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<SurfaceBoundaryFactory<ValueType>> clone() const override
    {
        return std::make_unique<Processor>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

}
