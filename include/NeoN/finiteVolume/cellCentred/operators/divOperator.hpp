// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @class Factory class to create divergence operators by a given name using
 * using NeoNs runTimeFactory mechanism
 */
template<typename ValueType>
class DivOperatorFactory :
    public RuntimeSelectionFactory<
        DivOperatorFactory<ValueType>,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{

public:

    static std::unique_ptr<DivOperatorFactory<ValueType>>
    create(const Executor& exec, const UnstructuredMesh& uMesh, const Input& inputs)
    {
        std::string key = (std::holds_alternative<Dictionary>(inputs))
                            ? std::get<Dictionary>(inputs).get<std::string>("DivOperator")
                            : std::get<TokenList>(inputs).next<std::string>();
        DivOperatorFactory<ValueType>::keyExistsOrError(key);
        return DivOperatorFactory<ValueType>::table().at(key)(exec, uMesh, inputs);
    }

    static std::string name() { return "DivOperatorFactory"; }

    DivOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh), sparsityPattern_(la::SparsityPattern::readOrCreate(mesh)) {};

    virtual ~DivOperatorFactory() {} // Virtual destructor

    // NOTE currently simple overloading is used here, because templating the virtual function
    // does not work and we cant template the entire class because the static create function
    // cannot access keyExistsOrError and table anymore.
    virtual void
    div(VolumeField<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const = 0;

    virtual void
    div(la::LinearSystem<ValueType, localIdx>& ls,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const = 0;

    virtual void
    div(Vector<ValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const = 0;

    virtual VolumeField<ValueType>
    div(const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& phi,
        const dsl::Coeff operatorScaling) const = 0;

    const la::SparsityPattern& getSparsityPattern() const { return sparsityPattern_; }

    // Pure virtual function for cloning
    virtual std::unique_ptr<DivOperatorFactory<ValueType>> clone() const = 0;

protected:

    const Executor exec_;

    const UnstructuredMesh& mesh_;

    const la::SparsityPattern& sparsityPattern_;
};

template<typename ValueType>
class DivOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    // copy constructor
    DivOperator(const DivOperator& divOp)
        : dsl::OperatorMixin<VolumeField<ValueType>>(
            divOp.exec_, divOp.coeffs_, divOp.field_, divOp.type_
        ),
          faceFlux_(divOp.faceFlux_),
          divOperatorStrategy_(
              divOp.divOperatorStrategy_ ? divOp.divOperatorStrategy_->clone() : nullptr
          ) {};

    DivOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        Input input
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          faceFlux_(faceFlux),
          divOperatorStrategy_(DivOperatorFactory<ValueType>::create(phi.exec(), phi.mesh(), input)
          ) {};

    DivOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi,
        std::unique_ptr<DivOperatorFactory<ValueType>> divOperatorStrategy
    )
        : dsl::OperatorMixin<VolumeField<scalar>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          faceFlux_(faceFlux), divOperatorStrategy_(std::move(divOperatorStrategy)) {};

    DivOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        VolumeField<ValueType>& phi
    )
        : dsl::OperatorMixin<VolumeField<ValueType>>(phi.exec(), dsl::Coeff(1.0), phi, termType),
          faceFlux_(faceFlux), divOperatorStrategy_(nullptr) {};


    void explicitOperation(Vector<ValueType>& source) const
    {
        NF_ASSERT(divOperatorStrategy_, "DivOperatorStrategy not initialized");
        auto tmpsource = Vector<ValueType>(source.exec(), source.size(), zero<ValueType>());
        const auto operatorScaling = this->getCoefficient();
        divOperatorStrategy_->div(tmpsource, faceFlux_, this->getVector(), operatorScaling);
        source += tmpsource;
    }

    la::LinearSystem<ValueType, localIdx> createEmptyLinearSystem() const
    {
        NF_ASSERT(divOperatorStrategy_, "DivOperatorStrategy not initialized");
        return divOperatorStrategy_->createEmptyLinearSystem();
    }

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) const
    {
        NF_ASSERT(divOperatorStrategy_, "DivOperatorStrategy not initialized");
        const auto operatorScaling = this->getCoefficient();
        divOperatorStrategy_->div(ls, faceFlux_, this->getVector(), operatorScaling);
    }

    void div(Vector<ValueType>& divPhi) const
    {
        const auto operatorScaling = this->getCoefficient();
        divOperatorStrategy_->div(divPhi, faceFlux_, this->getVector(), operatorScaling);
    }

    void div(la::LinearSystem<ValueType, localIdx>& ls) const
    {
        const auto operatorScaling = this->getCoefficient();
        divOperatorStrategy_->div(ls, faceFlux_, this->getVector(), operatorScaling);
    };

    void div(VolumeField<ValueType>& divPhi) const
    {
        const auto operatorScaling = this->getCoefficient();
        divOperatorStrategy_->div(divPhi, faceFlux_, this->getVector(), operatorScaling);
    }

    void read(const Input& input)
    {
        const UnstructuredMesh& mesh = this->getVector().mesh();
        if (std::holds_alternative<NeoN::Dictionary>(input))
        {
            auto dict = std::get<NeoN::Dictionary>(input);
            std::string schemeName = "div(" + faceFlux_.name + "," + this->getVector().name + ")";
            auto tokens = dict.subDict("divSchemes").get<NeoN::TokenList>(schemeName);
            divOperatorStrategy_ =
                DivOperatorFactory<ValueType>::create(this->exec(), mesh, tokens);
        }
        else
        {
            auto tokens = std::get<NeoN::TokenList>(input);
            divOperatorStrategy_ =
                DivOperatorFactory<ValueType>::create(this->exec(), mesh, tokens);
        }
    }

    std::string getName() const { return "DivOperator"; }

private:

    const SurfaceField<NeoN::scalar>& faceFlux_;

    std::unique_ptr<DivOperatorFactory<ValueType>> divOperatorStrategy_;
};


} // namespace NeoN
