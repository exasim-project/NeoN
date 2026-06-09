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

    // Per iteration: post the halo exchange. The proc-tail already holds the LOCAL face value
    // (written by the operator that produced the field — flux() / updateFaceVelocity() /
    // interpolation — or by constructFrom() at construction), so unlike the volume BC there is no
    // owner-cell -> face reconstruction to perform; we only ship the existing value to the
    // neighbour rank.
    virtual void update([[maybe_unused]] Field<ValueType>& domainVector) final
    {
#ifdef NF_WITH_MPI_SUPPORT
        fence(domainVector.exec());
        const int neighborRank =
            static_cast<int>(mesh_.boundaryMesh().neighbourRankForRange(this->range()));
        domainVector.boundaryData().communicate(this->range(), neighborRank);
#endif
    }

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
