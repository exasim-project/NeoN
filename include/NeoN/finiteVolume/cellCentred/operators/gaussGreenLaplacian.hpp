// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/laplacianOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"

namespace NeoN::finiteVolume::cellCentred
{

// Deferred non-orthogonal correction kernel for the Laplacian (defined in gaussGreenLaplacian.cpp).
// Declared here so other operators (e.g. GaussGreenDivLaplacian) can reuse it. Adds the explicit
// snGrad correction to the linear system's rhs; a no-op unless the snGrad scheme is corrected.
template<
    typename FieldValueType,
    typename AssemblyType = FieldValueType,
    typename SystemMatrixType = la::CSRMatrix<AssemblyType, localIdx>>
void computeLaplacianNonOrthCorrImpl(
    la::LinearSystem<AssemblyType, FieldValueType, SystemMatrixType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient
);

// Internal-face Laplacian assembly kernel (defined in gaussGreenLaplacian.cpp) -- the only piece
// of Laplacian assembly that touches upperIdx()/lowerIdx() as well as diagIdx(). Declared here,
// like computeLaplacianNonOrthCorrImpl above, so it can be instantiated for a SystemMatrixType
// other than the CSR default (e.g. ELL), independent of the still-CSR-only virtual laplacian()
// member below.
template<typename FieldValueType, typename AssemblyType, typename SystemMatrixType>
void computeLaplacianIntImpl(
    la::LinearSystem<AssemblyType, FieldValueType, SystemMatrixType>& ls,
    const SurfaceField<scalar>& gamma,
    const VolumeField<FieldValueType>& phi,
    const dsl::Coeff coeff,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient
);

template<typename FieldValueType, typename AssemblyType = FieldValueType>
class GaussGreenLaplacian :
    public LaplacianOperatorFactory<FieldValueType, AssemblyType>::template Register<
        GaussGreenLaplacian<FieldValueType, AssemblyType>>
{
    using Base = LaplacianOperatorFactory<FieldValueType, AssemblyType>::template Register<
        GaussGreenLaplacian<FieldValueType, AssemblyType>>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Laplacian"; }

    static std::string schema() { return "none"; }

    GaussGreenLaplacian(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolation_(exec, mesh, inputs),
          faceNormalGradient_(exec, mesh, inputs) {};

    virtual void laplacian(
        VolumeField<FieldValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff coeff
    ) override;

    virtual VolumeField<FieldValueType> laplacian(
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff coeff
    ) const override;

    virtual void laplacian(
        Vector<FieldValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff coeff
    ) override;

    virtual void laplacian(
        la::LinearSystem<AssemblyType, FieldValueType>& ls,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff coeff
    ) override;

    // ELL counterpart of the override above. Skips the CellBasedIterator dispatch (cell-based
    // ELL assembly is deferred) and shares the rest of the assembly sequence via laplacianImpl().
    virtual void laplacian(
        la::LinearSystem<AssemblyType, FieldValueType, la::ELLMatrix<AssemblyType, localIdx>>& ls,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff coeff
    ) override;

    std::unique_ptr<LaplacianOperatorFactory<FieldValueType, AssemblyType>> clone() const override
    {
        return std::make_unique<GaussGreenLaplacian<FieldValueType, AssemblyType>>(*this);
    };

private:

    SurfaceInterpolation<FieldValueType> surfaceInterpolation_;

    FaceNormalGradient<FieldValueType> faceNormalGradient_;

    // Shared by both laplacian(ls, ...) overrides above: face-to-matrix assembly,
    // boundary/proc-boundary contributions, and the non-orthogonal deferred correction. Defined
    // in gaussGreenLaplacian.cpp; not declared for cross-TU use since only the two overrides
    // call it.
    template<typename SystemMatrixType>
    void laplacianImpl(
        la::LinearSystem<AssemblyType, FieldValueType, SystemMatrixType>& ls,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff coeff
    );
};

// Required on MSVC: without extern template, each TU (DLL and EXE) gets its own
// instantiation of table() static local, so the DLL's addSubType() inserts into
// a different map than the one the test binary queries.
extern template class GaussGreenLaplacian<scalar>;
extern template class GaussGreenLaplacian<Vec3>;
extern template class GaussGreenLaplacian<Vec3, scalar>;

} // namespace NeoN
