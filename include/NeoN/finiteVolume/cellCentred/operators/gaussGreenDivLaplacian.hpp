// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenLaplacian.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/boundedDiv.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/linearAlgebra/meshIterationStrategies.hpp"

namespace NeoN::finiteVolume::cellCentred
{

// Fused div+laplacian internal-face, boundary, and proc-boundary assembly kernels (defined in
// gaussGreenDivLaplacian.cpp) -- declared here, like computeDivIntImp/computeLaplacianIntImpl in
// their own headers, so they can be instantiated for a SystemMatrixType other than the CSR
// default (e.g. ELL) and called from the header-inline ELL implicitOperation() overload below.
template<typename FieldValueType, typename AssemblyType, typename SystemMatrixType>
void computeDivLaplacianIntImpl(
    la::LinearSystem<AssemblyType, FieldValueType, SystemMatrixType>& ls,
    const VolumeField<FieldValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<FieldValueType>& divSurfInterp,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
);

template<typename FieldValueType, typename AssemblyType, typename SystemMatrixType>
void computeDivLaplacianBoundImpl(
    la::LinearSystem<AssemblyType, FieldValueType, SystemMatrixType>& ls,
    const VolumeField<FieldValueType>& u,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<FieldValueType>& divSurfInterp,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
);

template<typename FieldValueType, typename AssemblyType, typename SystemMatrixType>
void computeDivLaplacianProcBoundImpl(
    la::LinearSystem<AssemblyType, FieldValueType, SystemMatrixType>& ls,
    const VolumeField<FieldValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<FieldValueType>& divSurfInterp,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
);

// CSR-only cell-based counterpart of computeDivLaplacianIntImpl above (CellBasedIterator-driven,
// no SystemMatrixType parameter). Declared here rather than kept file-local so the header-inline
// implicitOperationImpl() below -- itself needed for ELL's type-erasure reachability -- can call
// it from its CSR branch regardless of which TU instantiates that branch.
template<typename FieldValueType, typename AssemblyType = FieldValueType>
void computeDivLaplacianIntCellBasedImpl(
    la::LinearSystem<AssemblyType, FieldValueType>& ls,
    const VolumeField<FieldValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<FieldValueType>& divSurfInterp,
    const FaceNormalGradient<FieldValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB
);

/* @brief
 *
 */
template<typename ValueType>
class GaussGreenDivLaplacian : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    GaussGreenDivLaplacian(const Executor& exec, Dictionary divConfig, Dictionary lapConfig);

    void explicitOperation(Vector<ValueType>& source) const;

    void implicitOperation(la::LinearSystem<ValueType>& ls) const;

    void implicitOperation(la::LinearSystem<scalar, ValueType>& ls) const
        requires(!std::is_same_v<ValueType, scalar>);

    // Format-generic overload, mirroring DivOperator/LaplacianOperator's
    // implicitOperation<SystemMatrixType> -- the non-template overloads above still win for the
    // CSR default, so existing DSL callers are unaffected; ELL callers deduce SystemMatrixType
    // from the ls argument. Header-inline (not gaussGreenDivLaplacian.cpp): OperatorModel<T>'s
    // vtable forces this to instantiate in whatever TU constructs a SpatialOperator from a
    // GaussGreenDivLaplacian -- e.g. DivLapOptimizer::optimize(), itself header-only -- so a
    // declare-in-header/define-in-cpp split would leave it undefined there (the same failure mode
    // DdtOperator hit before its equivalent methods were made header-inline).
    template<typename SystemMatrixType>
    void implicitOperation(la::LinearSystem<ValueType, ValueType, SystemMatrixType>& ls) const
    {
        implicitOperationImpl(ls);
    }

    // Format-generic segregated counterpart of the ELL overload above -- same header-inline
    // reasoning applies. implicitOperationImpl already handles AssemblyType != ValueType (the
    // non-template segregated CSR overload above already goes through it), so this is purely new
    // DSL-entry-point plumbing.
    template<typename SystemMatrixType>
        requires(!std::is_same_v<ValueType, scalar>)
    void implicitOperation(la::LinearSystem<scalar, ValueType, SystemMatrixType>& ls) const
    {
        implicitOperationImpl(ls);
    }

    void read(const Input& input);

    std::string getName() const;

    Dictionary getConfig() const;

private:

    dsl::Coeff coeffA_; // div coeff
    dsl::Coeff coeffB_; // lap coeff

    const SurfaceField<scalar>& gamma_;
    const SurfaceField<scalar>& flux_;

    std::shared_ptr<SurfaceInterpolation<ValueType>> divSurfaceInterpolation_;
    std::shared_ptr<FaceNormalGradient<ValueType>> faceNormalGradient_;

    // True when the div scheme carried a leading `bounded` prefix. The fused kernel then also emits
    // the bounded-convection Sp diagonal term (applyBoundedDivDiagonal), matching the un-fused
    // BoundedDiv path -- without it the momentum matrix loses its boundedness stabilisation and the
    // solve diverges.
    bool bounded_ = false;

    // Shared by the CSR entries (implicitOperation(la::LinearSystem<ValueType>&) and the
    // segregated form, both defined in gaussGreenDivLaplacian.cpp) and the ELL entry above, so
    // the assembly sequence (internal faces, non-orthogonal correction, physical/proc boundaries,
    // deferred div correction) has one source of truth instead of being duplicated per format.
    // Header-inline for the same reason as implicitOperation<SystemMatrixType> above -- it must
    // be instantiable wherever that public template is. Cell-based assembly stays CSR-only
    // (computeDivLaplacianIntCellBasedImpl is CSR-hardcoded); a CellBasedIterator on a non-CSR
    // system is rejected rather than silently falling back to face-based assembly.
    template<typename AssemblyType, typename SystemMatrixType>
    void implicitOperationImpl(la::LinearSystem<AssemblyType, ValueType, SystemMatrixType>& ls
    ) const
    {
        if constexpr (std::is_same_v<SystemMatrixType, la::CSRMatrix<AssemblyType, localIdx>>)
        {
            if (auto* cellIter =
                    dynamic_cast<la::CellBasedIterator*>(ls.getMeshIterator()->get().get()))
            {
                if (!cellIter->getCellBasedData())
                {
                    cellIter->setComputeCellBasedData(
                        this->getVector().mesh(), ls.matrix().sparsity(), ls.faceToMatrixAddress()
                    );
                }
                computeDivLaplacianIntCellBasedImpl(
                    ls,
                    this->getVector(),
                    flux_,
                    gamma_,
                    *divSurfaceInterpolation_,
                    *faceNormalGradient_,
                    coeffA_,
                    coeffB_
                );
            }
            else
            {
                computeDivLaplacianIntImpl(
                    ls,
                    this->getVector(),
                    flux_,
                    gamma_,
                    *divSurfaceInterpolation_,
                    *faceNormalGradient_,
                    coeffA_,
                    coeffB_
                );
            }
        }
        else
        {
            NF_ASSERT(
                dynamic_cast<la::CellBasedIterator*>(ls.getMeshIterator()->get().get()) == nullptr,
                "Cell-based iteration is not implemented for ELL assembly"
            );
            computeDivLaplacianIntImpl(
                ls,
                this->getVector(),
                flux_,
                gamma_,
                *divSurfaceInterpolation_,
                *faceNormalGradient_,
                coeffA_,
                coeffB_
            );
        }
        computeLaplacianNonOrthCorrImpl(
            ls, gamma_, this->getVector(), coeffB_, *faceNormalGradient_
        );
        computeDivLaplacianBoundImpl(
            ls,
            this->getVector(),
            flux_,
            gamma_,
            *divSurfaceInterpolation_,
            *faceNormalGradient_,
            coeffA_,
            coeffB_
        );
        computeDivLaplacianProcBoundImpl(
            ls,
            this->getVector(),
            flux_,
            gamma_,
            *divSurfaceInterpolation_,
            *faceNormalGradient_,
            coeffA_,
            coeffB_
        );

        // Deferred correction for a corrected div scheme (e.g. linearUpwind): the fused matrix
        // uses upwind div weights, the gradient correction is added explicitly to the rhs.
        if (divSurfaceInterpolation_->corrected())
        {
            const auto& mesh = this->getVector().mesh();
            SurfaceField<ValueType> correction(
                this->getVector().exec(),
                "divCorrection",
                mesh,
                createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh)
            );
            divSurfaceInterpolation_->correction(flux_, this->getVector(), correction);
            addDivCorrectionToRhs(ls, flux_, correction, coeffA_);
        }

        // Bounded-convection Sp diagonal term for a `bounded` div scheme (same as
        // BoundedDiv::div). applyBoundedDivDiagonal is CSR-only (no SystemMatrixType parameter of
        // its own); reject rather than silently skip the stabilisation term for ELL.
        if (bounded_)
        {
            if constexpr (std::is_same_v<SystemMatrixType, la::CSRMatrix<AssemblyType, localIdx>>)
            {
                applyBoundedDivDiagonal(ls, flux_, this->getVector().mesh(), coeffA_);
            }
            else
            {
                NF_ASSERT(
                    false,
                    "bounded div scheme is not yet implemented for ELL GaussGreenDivLaplacian"
                );
            }
        }
    }
};

// Required on MSVC: without extern template, each TU (DLL and EXE) gets its own
// instantiation, causing duplicate-symbol linker errors and bloating compile times.
extern template class GaussGreenDivLaplacian<scalar>;
extern template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN
