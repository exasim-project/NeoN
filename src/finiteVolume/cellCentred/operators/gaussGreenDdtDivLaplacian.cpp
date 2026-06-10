// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/error.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDdtDivLaplacian.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeDdtDivLapImplCell(
    la::LinearSystem<ValueType>& /*ls*/,
    scalar /*dt*/,
    const VolumeField<ValueType>& /*U*/,
    const SurfaceField<scalar>& /*phi*/,
    const SurfaceField<scalar>& /*gamma*/,
    const SurfaceInterpolation<ValueType>& /*divSurfInterp*/,
    const FaceNormalGradient<ValueType>& /*faceNormalGradient*/,
    const dsl::Coeff /*coeffA*/,
    const dsl::Coeff /*coeffB*/,
    std::shared_ptr<la::CellBasedIterator> /*iterator*/
)
{
    // TODO: implement cell-based fused ddt+div+lap assembly
    NF_ERROR_EXIT("computeDdtDivLapImplCell: not yet implemented");
}

template void
computeDdtDivLapImplCell<scalar>(la::LinearSystem<scalar>&, scalar, const VolumeField<scalar>&, const SurfaceField<scalar>&, const SurfaceField<scalar>&, const SurfaceInterpolation<scalar>&, const FaceNormalGradient<scalar>&, const dsl::Coeff, const dsl::Coeff, std::shared_ptr<la::CellBasedIterator>);

template void computeDdtDivLapImplCell<
    Vec3>(la::LinearSystem<Vec3>&, scalar, const VolumeField<Vec3>&, const SurfaceField<scalar>&, const SurfaceField<scalar>&, const SurfaceInterpolation<Vec3>&, const FaceNormalGradient<Vec3>&, const dsl::Coeff, const dsl::Coeff, std::shared_ptr<la::CellBasedIterator>);

} // namespace NeoN::finiteVolume::cellCentred
