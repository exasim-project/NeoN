// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeDivExp(
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const SurfaceInterpolation<ValueType>& surfInterp,
    Vector<ValueType>& divPhi,
    const dsl::Coeff operatorScaling
);

template<typename ValueType>
void computeDivImp(
    la::LinearSystem<ValueType, localIdx>& ls,
    const SurfaceField<scalar>& faceFlux,
    const VolumeField<ValueType>& phi,
    const dsl::Coeff operatorScaling,
    const SparsityPattern& sparsityPattern
);

template<typename ValueType>
class GaussGreenDiv :
    public DivOperatorFactory<ValueType>::template Register<GaussGreenDiv<ValueType>>
{
    using Base = DivOperatorFactory<ValueType>::template Register<GaussGreenDiv<ValueType>>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Divergence"; }

    static std::string schema() { return "none"; }

    GaussGreenDiv(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolation_(exec, mesh, inputs),
          sparsityPattern_(SparsityPattern::readOrCreate(mesh)) {};

    virtual void
    div(VolumeField<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override
    {
        computeDivExp<ValueType>(
            faceFlux, phi, surfaceInterpolation_, divPhi.internalVector(), operatorScaling
        );
    }

    virtual void
    div(la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override
    {
        computeDivImp(ls, faceFlux, phi, operatorScaling, *sparsityPattern_.get());
    };

    virtual void
    div(Vector<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override
    {
        computeDivExp<ValueType>(faceFlux, phi, surfaceInterpolation_, divPhi, operatorScaling);
    };

    virtual VolumeField<ValueType>
    div(const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const override
    {
        std::string name = "div(" + faceFlux.name + "," + phi.name + ")";
        VolumeField<ValueType> divPhi(
            this->exec_,
            name,
            this->mesh_,
            createCalculatedBCs<VolumeBoundary<ValueType>>(this->mesh_)
        );
        NeoN::fill(divPhi.internalVector(), zero<ValueType>());
        NeoN::fill(divPhi.boundaryData().value(), zero<ValueType>());
        computeDivExp<ValueType>(
            faceFlux, phi, surfaceInterpolation_, divPhi.internalVector(), operatorScaling
        );
        return divPhi;
    };

    std::unique_ptr<DivOperatorFactory<ValueType>> clone() const override
    {
        return std::make_unique<GaussGreenDiv<ValueType>>(*this);
    }

private:

    SurfaceInterpolation<ValueType> surfaceInterpolation_;

    // TODO why store as shared_ptr
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};

template class GaussGreenDiv<scalar>;
template class GaussGreenDiv<Vec3>;

} // namespace NeoN
