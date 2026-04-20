// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#ifdef NF_WITH_MPI_SUPPORT
#include <mpi.h>
#include "NeoN/core/mpi/environment.hpp"
#endif

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/mpi/operators.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

// TODO move to source file
namespace detail
{
// NOTE test with zero gradient first
// FIXME TODO exchange values on boundaries with neighbour rank
template<typename ValueType>
void setProcBoundaryValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    CommunicationPattern& commPattern
)
{
    const auto iVector = domainVector.internalVector().view();

    auto [refGradient, value, valueFraction, refValue, faceCells, deltaCoeffs] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceCells(),
        mesh.boundaryMesh().deltaCoeffs()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            refGradient[i] = zero<ValueType>();
            value[i] = iVector[faceCells[i]];
            valueFraction[i] = 0.0;          // only use refGrad
            refValue[i] = zero<ValueType>(); // not used
        },
        "setProcBoundaryValue"
    );
}
}

template<typename ValueType>
class Processor : public VolumeBoundaryFactory<ValueType>::template Register<Processor<ValueType>>
{
    using Base = VolumeBoundaryFactory<ValueType>::template Register<Processor<ValueType>>;

public:

    using ProcessorType = Processor<ValueType>;

    Processor(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = true}),
          nbrRank_(static_cast<int>(mesh.boundaryMesh().neighbourRank(
          )[static_cast<size_t>(patchID)]))
    {
        // store the local cell indices adjacent to this processor patch
        const auto faceCellsHost = mesh.boundaryMesh().faceCells().copyToHost();
        const auto faceCellsView = faceCellsHost.view();
        const localIdx start = this->start_;
        const localIdx end = this->end_;
        const auto patchSize = static_cast<std::size_t>(end - start);
        faceOwnerCells_.resize(patchSize);
        for (std::size_t i = 0; i < patchSize; ++i)
            faceOwnerCells_[i] = faceCellsView[start + static_cast<localIdx>(i)];
    }

    virtual void correctBoundaryCondition([[maybe_unused]] Field<ValueType>& domainVector) final
    {
#ifdef NF_WITH_MPI_SUPPORT
        if (nbrRank_ < 0) return;

        const localIdx patchSize = static_cast<localIdx>(faceOwnerCells_.size());
        if (patchSize == 0) return;

        // pack: gather internal cell values for the faces of this patch into a device vector
        Vector<localIdx> faceOwnerDev(domainVector.exec(), faceOwnerCells_.data(), patchSize);
        Vector<ValueType> sendDev(domainVector.exec(), patchSize);
        {
            const auto intView = domainVector.internalVector().view();
            const auto faceOwnerV = faceOwnerDev.view();
            auto sendView = sendDev.view();
            parallelFor(
                domainVector.exec(),
                {0, patchSize},
                NEON_LAMBDA(const localIdx i) { sendView[i] = intView[faceOwnerV[i]]; }
            );
        }
        fence(domainVector.exec());

        // copy packed values to host for MPI
        auto sendHost = sendDev.copyToHost();

        std::vector<ValueType> recvBuf(static_cast<std::size_t>(patchSize));
        mpi::Environment mpiEnv;
        MPI_Sendrecv(
            sendHost.data(),
            patchSize * static_cast<int>(sizeof(ValueType)),
            MPI_BYTE,
            nbrRank_,
            0,
            recvBuf.data(),
            patchSize * static_cast<int>(sizeof(ValueType)),
            MPI_BYTE,
            nbrRank_,
            0,
            mpiEnv.comm(),
            MPI_STATUS_IGNORE
        );

        // copy received values to device and write into the boundary data patch range
        Vector<ValueType> recvDev(domainVector.exec(), recvBuf.data(), patchSize);
        {
            const localIdx start = this->start_;
            auto bDataView = domainVector.boundaryData().value().view();
            const auto recvView = recvDev.view();
            parallelFor(
                domainVector.exec(),
                {0, patchSize},
                NEON_LAMBDA(const localIdx i) { bDataView[start + i] = recvView[i]; }
            );
        }
        fence(domainVector.exec());
#endif
    }

    static std::string name() { return "processor"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Processor>(*this);
    }

private:

    int nbrRank_ {-1};
    std::vector<localIdx> faceOwnerCells_;
};
}
