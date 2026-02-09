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
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    const SurfaceInterpolation<ValueType>& lapSurfInterp,
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
{

public:

    using VectorValueType = ValueType;

    GaussGreenDivLaplacian(const Executor& exec, Dictionary divConfig, Dictionary lapConfig)
        : dsl::OperatorMixin<VolumeField<ValueType>>(
            exec,
            dsl::Coeff(1.0),
            divConfig.get<VolumeField<ValueType>&>("field"),
            dsl::Operator::Type::Implicit
        ),
          sparsityPattern_(la::SparsityPattern::readOrCreate(this->getVector().mesh())),
          coeffA_(divConfig.get<dsl::Coeff>("coeff")), coeffB_(lapConfig.get<dsl::Coeff>("coeff")),
          gamma_(lapConfig.get<SurfaceField<scalar>&>("gamma")),
          flux_(divConfig.get<SurfaceField<scalar>&>("flux"))
    {
        // FIXME some sanity checks are needed
        // are div and lap field the same
    }

    void explicitOperation(Vector<ValueType>& source) const {}

    la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const {}

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) const
    {
        computeDivLapImpl(
            ls,
            this->getVector(),
            flux_,
            gamma_,
            *divSurfaceInterpolation_.get(),
            *lapSurfaceInterpolation_.get(),
            *faceNormalGradient_.get(),
            coeffA_,
            coeffB_,
            sparsityPattern_
        );
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
        lapSurfaceInterpolation_ =
            std::make_shared<SurfaceInterpolation<ValueType>>(this->exec(), mesh, laplTokens);
        divSurfaceInterpolation_ =
            std::make_shared<SurfaceInterpolation<ValueType>>(this->exec(), mesh, divTokens);
        faceNormalGradient_ =
            std::make_shared<FaceNormalGradient<ValueType>>(this->exec(), mesh, laplTokens);
    }

    std::string getName() const { return "FusedDivLapOperator"; }

    Dictionary getConfig() const { return {}; }

private:

    const la::SparsityPattern& sparsityPattern_;

    dsl::Coeff coeffA_; // div coeff
    dsl::Coeff coeffB_; // lap coeff

    const SurfaceField<scalar>& gamma_;
    const SurfaceField<scalar>& flux_;

    std::shared_ptr<SurfaceInterpolation<ValueType>> divSurfaceInterpolation_;
    std::shared_ptr<SurfaceInterpolation<ValueType>> lapSurfaceInterpolation_;
    std::shared_ptr<FaceNormalGradient<ValueType>> faceNormalGradient_;
};

template class GaussGreenDivLaplacian<scalar>;
template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN
