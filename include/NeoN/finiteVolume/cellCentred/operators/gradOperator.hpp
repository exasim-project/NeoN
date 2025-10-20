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

/* @class Factory class to create gradient operators by a given name using
 * using NeoNs runTimeFactory mechanism
 */
template<typename ValueType>
class GradOperatorFactory :
    public RuntimeSelectionFactory<
        GradOperatorFactory<ValueType>,
        Parameters<const Executor&, const UnstructuredMesh&>>
{

public:

    static std::unique_ptr<GradOperatorFactory<ValueType>>
    create(const Executor& exec, const UnstructuredMesh& uMesh, const Input& inputs)
    {
        std::string key = (std::holds_alternative<Dictionary>(inputs))
                            ? std::get<Dictionary>(inputs).get<std::string>("GradOperator")
                            : std::get<TokenList>(inputs).next<std::string>();
        GradOperatorFactory<ValueType>::keyExistsOrError(key);
        return GradOperatorFactory<ValueType>::table().at(key)(exec, uMesh);
    }

    static std::string name() { return "GradOperatorFactory"; }

    GradOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh), sparsityPattern_(la::SparsityPattern::readOrCreate(mesh)) {};

    virtual ~GradOperatorFactory() = default; // Virtual destructor

    /* @brief compute implicit gradient operator contribution
     *
     * @param [in] phi
     * @param [in] operatorScaling
     * @param [in,out] ls the linear system to assemble into
     */
    virtual void grad(
        const VolumeField<scalar>& phi,
        const dsl::Coeff operatorScaling,
        la::LinearSystem<ValueType, localIdx>& ls
    ) const = 0;

    /* @brief compute explicit gradient operator
     *
     * @param phi [in] - field for which the gradient is computed
     * @param operatorScaling [in] - scales operator by a coefficient
     * @param gradPhi [in,out] - resulting gradient field
     */
    virtual void grad(
        const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling, Vector<Vec3>& gradPhi
    ) const = 0;

    /* @brief compute explicit gradient operator and return result
     *
     * @param phi [in] - field for which the gradient is computed
     * @param operatorScaling [in] - scales operator by a coefficient
     * @return gradPhi - resulting gradient field
     */
    virtual VolumeField<ValueType>
    grad(const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling) const = 0;

    [[deprecated("This function will be removed")]] const la::SparsityPattern&
    getSparsityPattern() const
    {
        return sparsityPattern_;
    }

    // Pure virtual function for cloning
    virtual std::unique_ptr<GradOperatorFactory<ValueType>> clone() const = 0;

protected:

    const Executor exec_;

    const UnstructuredMesh& mesh_;

    const la::SparsityPattern& sparsityPattern_;
};

template<typename ValueType>
class GradOperator : public dsl::OperatorMixin<VolumeField<ValueType>, VolumeField<scalar>>
{

public:

    using VectorValueType = ValueType;

    // copy constructor
    GradOperator(const GradOperator& gradOp)
        : dsl::OperatorMixin<VolumeField<ValueType>, VolumeField<scalar>>(
            gradOp.exec_, gradOp.coeffs_, gradOp.field_, gradOp.type_
        ),
          gradOperatorStrategy_(
              gradOp.gradOperatorStrategy_ ? gradOp.gradOperatorStrategy_->clone() : nullptr
          ) {};

    GradOperator(dsl::Operator::Type termType, const VolumeField<scalar>& phi, const Input& input)
        : dsl::OperatorMixin<VolumeField<ValueType>, VolumeField<scalar>>(
            phi.exec(), dsl::Coeff(1.0), phi, termType
        ),
          gradOperatorStrategy_(
              GradOperatorFactory<ValueType>::create(phi.exec(), phi.mesh(), input)
          ) {};

    GradOperator(
        dsl::Operator::Type termType,
        const VolumeField<scalar>& phi,
        std::unique_ptr<GradOperatorFactory<ValueType>> gradOperatorStrategy
    )
        : dsl::OperatorMixin<VolumeField<ValueType>, VolumeField<scalar>>(
            phi.exec(), dsl::Coeff(1.0), phi, termType
        ),
          gradOperatorStrategy_(std::move(gradOperatorStrategy)) {};

    GradOperator(dsl::Operator::Type termType, const VolumeField<scalar>& phi)
        : dsl::OperatorMixin<VolumeField<ValueType>, VolumeField<scalar>>(
            phi.exec(), dsl::Coeff(1.0), phi, termType
        ),
          gradOperatorStrategy_(nullptr) {};


    void explicitOperation(Vector<Vec3>& source) const
    {
        NF_ASSERT(gradOperatorStrategy_, "GradOperatorStrategy not initialized");
        auto tmpsource = Vector<Vec3>(source.exec(), source.size(), zero<Vec3>());
        const auto operatorScaling = this->getCoefficient();
        gradOperatorStrategy_->grad(this->getVector(), operatorScaling, tmpsource);
        source += tmpsource;
    }

    [[deprecated("This function will be removed")]] la::LinearSystem<ValueType, localIdx>
    createEmptyLinearSystem() const
    {
        NF_ASSERT(gradOperatorStrategy_, "GradOperatorStrategy not initialized");
        return gradOperatorStrategy_->createEmptyLinearSystem();
    }

    /* @brief forwards to implicit gradOperatorStrategy_->grad() with arguments */
    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) const
    {
        NF_ASSERT(gradOperatorStrategy_, "GradOperatorStrategy not initialized");
        const auto operatorScaling = this->getCoefficient();
        gradOperatorStrategy_->grad(this->getVector(), operatorScaling, ls);
    }

    /* @brief forwards to  gradOperatorStrategy_->grad() with arguments */
    [[deprecated("use explicit or implicit operation")]] void grad(auto&&... args) const
    {
        const auto operatorScaling = this->getCoefficient();
        gradOperatorStrategy_->grad(
            std::forward<decltype(args)>(args)..., this->getVector(), operatorScaling
        );
    }

    void read(const Input& input)
    {
        const UnstructuredMesh& mesh = this->getVector().mesh();
        if (std::holds_alternative<NeoN::Dictionary>(input))
        {
            auto dict = std::get<NeoN::Dictionary>(input);
            std::string schemeName = "grad(" + this->getVector().name + ")";
            auto tokens = dict.subDict("gradSchemes").get<NeoN::TokenList>(schemeName);
            gradOperatorStrategy_ =
                GradOperatorFactory<ValueType>::create(this->exec(), mesh, tokens);
        }
        else
        {
            auto tokens = std::get<NeoN::TokenList>(input);
            gradOperatorStrategy_ =
                GradOperatorFactory<ValueType>::create(this->exec(), mesh, tokens);
        }
    }

    std::string getName() const { return "GradOperator"; }

private:

    std::unique_ptr<GradOperatorFactory<ValueType>> gradOperatorStrategy_;
};


} // namespace NeoN
