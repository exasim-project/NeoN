// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <memory>

#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/core/primitives/symmTensor.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLinearInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst
)
{
    NeoN::mpi::Environment mpiEnviron;
    const auto exec = dst.exec();
    auto dstS = dst.internalVector().view();
    // Also populate dst.boundaryData().value(): callers that iterate the
    // SurfaceField's per-patch boundary view (e.g. nf::compare with
    // withBoundaries=true, or any operator that consumes the boundary
    // representation of an interpolated surface field) need the boundary
    // half to carry the same value the kernel writes into internalVector
    // at boundary face indices.
    auto dstBnd = dst.boundaryData().value().view();
    const auto [srcS, weightS, ownerS, neighS, boundS, bcFaceCells] = views(
        src.internalVector(),
        weights.internalVector(),
        dst.mesh().faceOwner(),
        dst.mesh().faceNeighbour(),
        dst.mesh().boundaryMesh().faceCells(),
        src.boundaryData().value()
    );

    auto nInternalFaces = src.mesh().nInternalFaces();
    auto nBoundaryFaces = src.mesh().nBoundaryFaces();
    auto totalFaces = src.mesh().nTotalFaces();

    NF_ASSERT(dstS.size() == totalFaces, "Inconsistent size");
    //    NF_ASSERT(dstS.size() == ownerS.size(), "Inconsistent size");
    // NF_ASSERT(dstS.size() == neighS.size(), "Inconsistent size");
    NF_ASSERT(dstS.size() == weightS.size(), "Inconsistent size");

    NeoN::parallelFor(
        exec,
        {0, dstS.size()},
        NEON_LAMBDA(const localIdx facei) {
            if (facei < nInternalFaces)
            {
                auto own = ownerS[facei];
                auto nei = neighS[facei];
                dstS[facei] = weightS[facei] * srcS[own] + (1.0 - weightS[facei]) * srcS[nei];
            }
            // regular boundary
            if (facei >= nInternalFaces && facei < nInternalFaces + nBoundaryFaces)
            {
                auto bcfacei = facei - nInternalFaces;
                dstS[facei] = weightS[facei] * boundS[bcfacei];
                dstBnd[bcfacei] = dstS[facei];
            }
            // proc boundary
            //
            // Same linear blend as an internal face, but the neighbour cell
            // lives on another rank. After VolumeField::correctBoundaryConditions
            // the exchanged neighbour value sits in boundS[bcfacei].
            //
            // Use bm.faceCells() (compressed indexing matching the boundary tail
            // of the surface field) for the owner cell. mesh.faceOwner() is in
            // OpenFOAM's full face indexing which includes empty patches;
            // indexing it with the compressed proc-face index reads the wrong
            // (empty-patch) face's owner.
            if (facei >= nInternalFaces + nBoundaryFaces && facei < totalFaces)
            {
                auto bcfacei = facei - nInternalFaces;
                auto own = faceCellS[bcfacei];
                dstS[facei] = (1 - weightS[facei]) * srcS[own] + weightS[facei] * boundS[bcfacei];
                dstBnd[bcfacei] = dstS[facei];
            }
        },
        "computeLinearInterpolation"
    );
}

#define NF_DECLARE_COMPUTE_IMP_LIN_INT(TYPENAME)                                                   \
    template void computeLinearInterpolation<                                                      \
        TYPENAME>(const VolumeField<TYPENAME>&, const SurfaceField<scalar>&, SurfaceField<TYPENAME>&)

NF_DECLARE_COMPUTE_IMP_LIN_INT(scalar);
NF_DECLARE_COMPUTE_IMP_LIN_INT(Vec3);
NF_DECLARE_COMPUTE_IMP_LIN_INT(Tensor);
NF_DECLARE_COMPUTE_IMP_LIN_INT(SymmTensor);

// template class Linear<scalar>;
// template class Linear<Vec3>;

} // namespace NeoN
