// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/distributed/communicationPattern.hpp"
#endif

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{
// One-time (set): zero the unused mixed-BC coefficients (refGrad / valueFraction / refValue) on
// the processor patch. These are constant for the lifetime of the field, so this runs once.
template<typename ValueType>
void setProcBoundaryCoefficients(
    Field<ValueType>& domainVector, std::pair<localIdx, localIdx> range
);

// Per-iteration (update): seed the proc-patch ghost with the owner-cell value
// (value[i] = internalVector[faceCells[i]]) so it can be shipped to the neighbour rank.
template<typename ValueType>
void updateProcBoundaryOwnerValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
);

extern template void
setProcBoundaryCoefficients<scalar>(Field<scalar>&, std::pair<localIdx, localIdx>);
extern template void setProcBoundaryCoefficients<Vec3>(Field<Vec3>&, std::pair<localIdx, localIdx>);
extern template void
setProcBoundaryCoefficients<Tensor>(Field<Tensor>&, std::pair<localIdx, localIdx>);

extern template void updateProcBoundaryOwnerValue<
    scalar>(Field<scalar>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);
extern template void updateProcBoundaryOwnerValue<
    Vec3>(Field<Vec3>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);
extern template void updateProcBoundaryOwnerValue<
    Tensor>(Field<Tensor>&, const UnstructuredMesh&, std::pair<localIdx, localIdx>);
}

template<typename ValueType>
class Processor : public VolumeBoundaryFactory<ValueType>::template Register<Processor<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<Processor<ValueType>>;

public:

    using ProcessorType = Processor<ValueType>;

    Processor(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true, .fixesValue = false}), mesh_(mesh)
    {}

    // One-time initialisation: the unused mixed-BC coefficients never change after construction.
    virtual void set(Field<ValueType>& domainVector) final
    {
        detail::setProcBoundaryCoefficients(domainVector, this->range());
    }

    // Per iteration: re-seed the owner value into the proc-patch ghost and stage the halo exchange.
    // The actual MPI isend/irecv is deferred to waitAll() (triggered by the next value() call
    // outside the correctBoundaryConditions loop) to preserve "post all, drain once" discipline.
    //
    // Phase 14 OVERLAP-01 note: Option B (extracting post-all to
    // VolumeField::correctBoundaryConditions with start/finish split) is the migration path for
    // explicit compute/comm overlap — not here.
    virtual void update([[maybe_unused]] Field<ValueType>& domainVector) final
    {
        detail::updateProcBoundaryOwnerValue(domainVector, mesh_, this->range());
#ifdef NF_WITH_MPI_SUPPORT
        // LOAD-BEARING FENCE (COMM-03): updateProcBoundaryOwnerValue above launches a GPU
        // kernel writing the owner-cell value into the proc-patch ghost (device memory at
        // value_.data() + rangeStart). communicate() stages values that waitAll() will
        // send via MPI_Isend reading that same device pointer; fencing here ensures the
        // kernel has landed before MPI reads device memory in waitAll().
        fence(domainVector.exec());
        const int neighborRank =
            static_cast<int>(mesh_.boundaryMesh().neighbourRankForRange(this->range()));
        const auto& pattern = cachedCommunicationPattern(mesh_);
        // mesh_.nBoundaryFaces() returns the count of PHYSICAL (non-proc) boundary faces.
        // value_ layout: [physical_faces..., proc_faces...] so physical-face count = proc start
        // index.
        const localIdx procFaceStart = mesh_.nBoundaryFaces();
        domainVector.boundaryData().communicate(
            this->range(), neighborRank, pattern, procFaceStart
        );
#endif
    }

    // Full correction = one-time set() + per-iteration update(). Kept for the BoundaryContext
    // path and any direct caller; the field's correctBoundaryConditions() calls set()/update()
    // separately so the coefficient kernel runs only once.
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

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Processor>(*this);
    }

    std::string getName() const override { return name(); }


private:

    const UnstructuredMesh& mesh_;
};
}
