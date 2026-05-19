// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include <mpi.h>
#include <Kokkos_Core.hpp>

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
    std::pair<localIdx, localIdx> range
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

    if (std::getenv("NF_PROC_BC_TRACE"))
    {
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        // Pull the data we just wrote back to host so we can print it.
        auto valueH = domainVector.boundaryData().value().copyToHost();
        auto faceCellsH = mesh.boundaryMesh().faceCells().copyToHost();
        auto iVecH = domainVector.internalVector().copyToHost();
        const auto vV = valueH.view();
        const auto fV = faceCellsH.view();
        const auto iV = iVecH.view();
        const localIdx nCells = static_cast<localIdx>(iVecH.size());
        std::fprintf(
            stderr,
            "[NF_PROC_BC_TRACE][rank %d][setProcBoundaryValue] range=[%lld,%lld) "
            "iVec.size=%lld\n",
            r,
            (long long)range.first,
            (long long)range.second,
            (long long)nCells
        );
        for (localIdx i = range.first; i < range.second; ++i)
        {
            const auto fc = fV[i];
            const bool oob = (fc < 0 || fc >= nCells);
            // Print first 4 entries + always print OOB
            if (i - range.first < 4 || oob)
            {
                if constexpr (std::is_same_v<ValueType, NeoN::scalar>)
                {
                    std::fprintf(
                        stderr,
                        "[NF_PROC_BC_TRACE][rank %d][setProcBoundaryValue]   "
                        "i=%lld faceCells[i]=%lld %s iVec[faceCells]=%.6e value[i]=%.6e\n",
                        r,
                        (long long)i,
                        (long long)fc,
                        oob ? "OOB!" : "",
                        oob ? 0.0 : (double)iV[fc],
                        (double)vV[i]
                    );
                }
                else
                {
                    std::fprintf(
                        stderr,
                        "[NF_PROC_BC_TRACE][rank %d][setProcBoundaryValue]   "
                        "i=%lld faceCells[i]=%lld %s (Vec3)\n",
                        r,
                        (long long)i,
                        (long long)fc,
                        oob ? "OOB!" : ""
                    );
                }
            }
        }
    }
}
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

    virtual void correctBoundaryCondition([[maybe_unused]] Field<ValueType>& domainVector) final
    {
        detail::setProcBoundaryValue(domainVector, mesh_, this->range());
    }

    static std::string name() { return "processor"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Processor>(*this);
    }

    virtual std::string getName() const { return name(); }


private:

    const UnstructuredMesh& mesh_;
};
}
