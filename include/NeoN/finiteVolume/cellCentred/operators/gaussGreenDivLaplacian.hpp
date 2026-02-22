// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/divOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeDivLapImplCell(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    // const SurfaceInterpolation<ValueType>& lapSurfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
    const dsl::Coeff coeffA,
    const dsl::Coeff coeffB,
    std::shared_ptr<la::CellBasedIterator> iterator
);

template<typename ValueType>
void computeDivLapImplFace(
    la::LinearSystem<ValueType>& ls,
    const VolumeField<ValueType>& U,
    const SurfaceField<scalar>& phi,
    const SurfaceField<scalar>& gamma,
    const SurfaceInterpolation<ValueType>& divSurfInterp,
    // const SurfaceInterpolation<ValueType>& lapSurfInterp,
    const FaceNormalGradient<ValueType>& faceNormalGradient,
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

    GaussGreenDivLaplacian(const Executor& exec, Dictionary divConfig, Dictionary lapConfig)
        : dsl::OperatorMixin<VolumeField<ValueType>>(
            exec,
            dsl::Coeff(1.0),
            divConfig.get<detail::RefHolder<VolumeField<ValueType>>>("field").c,
            dsl::Operator::Type::Implicit
        ),
          coeffA_(divConfig.get<detail::RefHolder<dsl::Coeff>>("coeff").c),
          coeffB_(lapConfig.get<detail::RefHolder<dsl::Coeff>>("coeff").c),
          gamma_(lapConfig.get<detail::RefHolder<SurfaceField<scalar>>>("gamma").c),
          flux_(divConfig.get<detail::RefHolder<SurfaceField<scalar>>>("flux").c)
    {
        // FIXME some sanity checks are needed
        // are div and lap field the same
    }

    void explicitOperation(Vector<ValueType>& source) const {}

    void implicitOperation(la::LinearSystem<ValueType>& ls) const
    {
        // FIXME I dont know how we can end up with a nullptr here double check
        if (ls.getMeshIterator() == nullptr)
        {
            computeDivLapImplFace(
                ls,
                this->getVector(),
                flux_,
                gamma_,
                *divSurfaceInterpolation_.get(),
                //    *lapSurfaceInterpolation_.get(),
                *faceNormalGradient_.get(),
                coeffA_,
                coeffB_
            );
            return;
        }

        if (ls.getMeshIterator()->name() == "CellBased")
        {
            computeDivLapImplCell(
                ls,
                this->getVector(),
                flux_,
                gamma_,
                *divSurfaceInterpolation_.get(),
                //    *lapSurfaceInterpolation_.get(),
                *faceNormalGradient_.get(),
                coeffA_,
                coeffB_,
                std::dynamic_pointer_cast<la::CellBasedIterator>(ls.getMeshIterator()->get())
            );
            return;
        }
        if (ls.getMeshIterator()->name() == "FaceBased")
        {
            computeDivLapImplFace(
                ls,
                this->getVector(),
                flux_,
                gamma_,
                *divSurfaceInterpolation_.get(),
                //    *lapSurfaceInterpolation_.get(),
                *faceNormalGradient_.get(),
                coeffA_,
                coeffB_
            );
            return;
        }
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
        laplTokens.remove(0);
        divTokens.remove(0);
        //       lapSurfaceInterpolation_ =
        //           std::make_shared<SurfaceInterpolation<ValueType>>(this->exec(), mesh,
        //           laplTokens);
        divSurfaceInterpolation_ =
            std::make_shared<SurfaceInterpolation<ValueType>>(this->field_.exec(), mesh, divTokens);
        laplTokens.remove(0);
        faceNormalGradient_ =
            std::make_shared<FaceNormalGradient<ValueType>>(this->field_.exec(), mesh, laplTokens);
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
    // std::shared_ptr<SurfaceInterpolation<scalar>> lapSurfaceInterpolation_;
    std::shared_ptr<FaceNormalGradient<ValueType>> faceNormalGradient_;
};

template class GaussGreenDivLaplacian<scalar>;
template class GaussGreenDivLaplacian<Vec3>;

} // namespace NeoN
