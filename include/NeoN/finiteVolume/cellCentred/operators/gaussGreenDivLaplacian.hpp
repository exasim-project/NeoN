// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"

namespace NeoN::finiteVolume::cellCentred
{


template<typename ValueType>
void computeDivLapImpl(
    la::LinearSystem<ValueType, localIdx>& ls,
    const VolumeField<ValueType>& phi,
    const SurfaceField<scalar>& faceFlux,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& surfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB,
    const la::SparsityPattern& sp
);

/* @brief
 *
 */
template<typename ValueType>
class GaussGreenDivLaplacian : public dsl::OperatorMixin<VolumeField<ValueType>>
// public DivOperatorFactory<ValueType>::template Register<GaussGreenDivLaplacian<ValueType>>
{
    // using Base =
    //     DivOperatorFactory<ValueType>::template Register<GaussGreenDivLaplacian<ValueType>>;

public:

    using VectorValueType = ValueType;

    GaussGreenDivLaplacian(const Executor& exec, Dictionary divConfig, Dictionary lapConfig)
        : dsl::OperatorMixin<VolumeField<ValueType>>(
            exec,
            dsl::Coeff(1.0),
            divConfig.get<VolumeField<ValueType>&>("field"),
            dsl::Operator::Type::Implicit
        ),
          coeffA_(divConfig.get<dsl::Coeff>("coeff")), coeffB_(lapConfig.get<dsl::Coeff>("coeff")),
          gamma_(lapConfig.get<SurfaceField<scalar>&>("gamma")),
          flux_(divConfig.get<SurfaceField<scalar>&>("flux"))
    {
        // FIXME some sanity checks are needed
        // are div and lap field the same
    }

    // ,
    //   surfaceInterpolation_(exec, mesh, inputs) {};

    // std::unique_ptr<DivOperatorFactory<ValueType>> clone() const
    // {
    //     return std::make_unique<GaussGreenDivLaplacian<ValueType>>(*this);
    // }

    void explicitOperation(Vector<ValueType>& source) const {}

    la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const {}

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) const
    {
        //  computeDivLapImpl(
        //     ls,
        //     const VolumeField<ValueType>& phi,
        //      face_,
        //      gamma_,
        //     laplSurfaceInterpolation_,
        //     divSurfaceInterpolation_,
        //     faceNormalGradient_,
        //      coeffA_,
        //      coeffB_,
        //     const la::SparsityPattern& sp
        // );
    }

    void read(const Input& input)
    {
        const UnstructuredMesh& mesh = this->field_.mesh();
        TokenList laplTokens;
        TokenList divTokens;
        if (std::holds_alternative<Dictionary>(input))
        {
            auto dict = std::get<Dictionary>(input);
            std::string lapSchemeName = "laplacian(" + gamma_.name + "," + this->field_.name + ")";
            std::string divSchemeName = "div(" + flux_.name + "," + this->getVector().name + ")";
            laplTokens = dict.subDict("laplacianSchemes").get<NeoN::TokenList>(lapSchemeName);
            divTokens = dict.subDict("divSchemes").get<NeoN::TokenList>(divSchemeName);
        }
        else
        {
            NF_ERROR_EXIT("only dictionary input supported");
        }
        laplSurfaceInterpolation_ =
            std::make_shared<SurfaceInterpolation<ValueType>>(this->exec(), mesh, laplTokens);
        divSurfaceInterpolation_ =
            std::make_shared<SurfaceInterpolation<ValueType>>(this->exec(), mesh, divTokens);
        faceNormalGradient_ =
            std::make_shared<FaceNormalGradient<ValueType>>(this->exec(), mesh, laplTokens);
    }

    std::string getName() const { return "FusedDivLapOperator"; }

    Dictionary getConfig() const { return {}; }

private:

    // SurfaceInterpolation<ValueType> surfaceInterpolation_;

    dsl::Coeff coeffA_; // div coeff
    dsl::Coeff coeffB_; // lap coeff

    const SurfaceField<scalar>& gamma_;
    const SurfaceField<scalar>& flux_;

    std::shared_ptr<SurfaceInterpolation<ValueType>> divSurfaceInterpolation_;
    std::shared_ptr<SurfaceInterpolation<ValueType>> laplSurfaceInterpolation_;
    std::shared_ptr<FaceNormalGradient<ValueType>> faceNormalGradient_;
};

template class GaussGreenDivLaplacian<scalar>;
template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN
