// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred
{

// Factory base for fused convection–diffusion
template<typename ValueType>
class ConvDiffOperatorFactory :
    public RuntimeSelectionFactory<
        ConvDiffOperatorFactory<ValueType>,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{

public:

    static std::unique_ptr<ConvDiffOperatorFactory<ValueType>>
    create(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
    {
        std::string key = (std::holds_alternative<Dictionary>(inputs))
                            ? std::get<Dictionary>(inputs).get<std::string>("ConvDiffOperator")
                            : std::get<TokenList>(inputs).next<std::string>();
        ConvDiffOperatorFactory<ValueType>::keyExistsOrError(key);
        return ConvDiffOperatorFactory<ValueType>::table().at(key)(exec, mesh, inputs);
    }

    static std::string name() { return "ConvDiffOperatorFactory"; }

    ConvDiffOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh), sparsityPattern_(la::SparsityPattern::readOrCreate(mesh)) {};

    virtual ~ConvDiffOperatorFactory() {}

    // explicit assembly into VolumeField
    virtual void convDiff(
        VolumeField<ValueType>& result,
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) = 0;

    // explicit: return-by-value convenience
    virtual VolumeField<ValueType> convDiff(
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) const = 0;

    // explicit assembly into a vector (source term)
    virtual void convDiff(
        Vector<ValueType>& result,
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) = 0;

    // implicit assembly into linear system
    virtual void convDiff(
        la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling
    ) = 0;

    // cloning
    virtual std::unique_ptr<ConvDiffOperatorFactory<ValueType>> clone() const = 0;

    [[deprecated("This function will be removed")]] const la::SparsityPattern&
    getSparsityPattern() const
    {
        return sparsityPattern_;
    }

protected:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    const la::SparsityPattern& sparsityPattern_;
};


// DSL-level fused operator wrapper (similar to DivOperator / LaplacianOperator)
template<typename ValueType>
class ConvDiffOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{
public:

    using VectorValueType = ValueType;

    // copy ctor
    ConvDiffOperator(const ConvDiffOperator& cdOp)
        : dsl::OperatorMixin<VolumeField<ValueType>>(
            cdOp.exec_, cdOp.coeffs_, cdOp.field_, cdOp.type_
        ),
          faceFlux_(cdOp.faceFlux_), gamma_(cdOp.gamma_),
          convDiffOperatorStrategy_(
              cdOp.convDiffOperatorStrategy_ ? cdOp.convDiffOperatorStrategy_->clone() : nullptr
          )
    {}

    ConvDiffOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        Input input
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          faceFlux_(faceFlux), gamma_(gamma),
          convDiffOperatorStrategy_(
              ConvDiffOperatorFactory<ValueType>::create(this->exec_, phi.mesh(), input)
          )
    {}

    ConvDiffOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi,
        std::unique_ptr<ConvDiffOperatorFactory<ValueType>> convDiffOperatorStrategy
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          faceFlux_(faceFlux), gamma_(gamma),
          convDiffOperatorStrategy_(std::move(convDiffOperatorStrategy))
    {}

    ConvDiffOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        const SurfaceField<scalar>& gamma,
        VolumeField<ValueType>& phi
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          faceFlux_(faceFlux), gamma_(gamma), convDiffOperatorStrategy_(nullptr)
    {}

    // explicit: add to source
    void explicitOperation(Vector<ValueType>& source) const
    {
        NF_ASSERT(convDiffOperatorStrategy_, "ConvDiffOperatorStrategy not initialized");
        auto tmpsource = Vector<ValueType>(source.exec(), source.size(), zero<ValueType>());
        const auto operatorScaling = this->getCoefficient();
        convDiffOperatorStrategy_->convDiff(
            tmpsource, faceFlux_, gamma_, this->field_, operatorScaling
        );
        source += tmpsource;
    }

    // implicit: add to matrix
    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) const
    {
        NF_ASSERT(convDiffOperatorStrategy_, "ConvDiffOperatorStrategy not initialized");
        const auto operatorScaling = this->getCoefficient();
        convDiffOperatorStrategy_->convDiff(ls, faceFlux_, gamma_, this->field_, operatorScaling);
    }

    [[deprecated("use explicit or implicit operation")]] void convDiff(auto&&... args) const
    {
        const auto operatorScaling = this->getCoefficient();
        convDiffOperatorStrategy_->convDiff(
            std::forward<decltype(args)>(args)..., faceFlux_, gamma_, this->field_, operatorScaling
        );
    }

    void read(const Input& input)
    {
        const UnstructuredMesh& mesh = this->field_.mesh();
        if (std::holds_alternative<NeoN::Dictionary>(input))
        {
            auto dict = std::get<NeoN::Dictionary>(input);
            std::string schemeName =
                "convDiff(" + faceFlux_.name + "," + gamma_.name + "," + this->field_.name + ")";
            auto tokens = dict.subDict("convDiffSchemes").get<NeoN::TokenList>(schemeName);
            convDiffOperatorStrategy_ =
                ConvDiffOperatorFactory<ValueType>::create(this->exec(), mesh, tokens);
        }
        else
        {
            auto tokens = std::get<NeoN::TokenList>(input);
            convDiffOperatorStrategy_ =
                ConvDiffOperatorFactory<ValueType>::create(this->exec(), mesh, tokens);
        }
    }

    std::string getName() const { return "ConvDiffOperator"; }

private:

    const SurfaceField<scalar>& faceFlux_;
    const SurfaceField<scalar>& gamma_;

    std::unique_ptr<ConvDiffOperatorFactory<ValueType>> convDiffOperatorStrategy_;
};

} // namespace NeoN::finiteVolume::cellCentred
