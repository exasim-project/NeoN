// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
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

/* @class Factory class to create laplacian operators by a given name using
 * using NeoNs runTimeFactory mechanism
 */
template<typename FieldValueType, typename AssemblyType = FieldValueType>
class LaplacianOperatorFactory :
    public RuntimeSelectionFactory<
        LaplacianOperatorFactory<FieldValueType, AssemblyType>,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{

public:

    static std::unique_ptr<LaplacianOperatorFactory<FieldValueType, AssemblyType>>
    create(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
    {
        std::string key = (std::holds_alternative<Dictionary>(inputs))
                            ? std::get<Dictionary>(inputs).get<std::string>("LaplacianOperator")
                            : std::get<TokenList>(inputs).next<std::string>();
        LaplacianOperatorFactory<FieldValueType, AssemblyType>::keyExistsOrError(key);
        return LaplacianOperatorFactory<FieldValueType, AssemblyType>::table().at(key)(
            exec, mesh, inputs
        );
    }

    static std::string name() { return "LaplacianOperatorFactory"; }

    LaplacianOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~LaplacianOperatorFactory() {} // Virtual destructor

    virtual void laplacian(
        VolumeField<FieldValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling
    ) = 0;

    virtual VolumeField<FieldValueType> laplacian(
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling
    ) const = 0;

    virtual void laplacian(
        Vector<FieldValueType>& lapPhi,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling
    ) = 0;

    virtual void laplacian(
        la::LinearSystem<AssemblyType, FieldValueType>& ls,
        const SurfaceField<scalar>& gamma,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling
    ) = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<LaplacianOperatorFactory<FieldValueType, AssemblyType>>
    clone() const = 0;

protected:

    const Executor exec_;

    const UnstructuredMesh& mesh_;
};

template<typename FieldValueType>
class LaplacianOperator : public dsl::OperatorMixin<VolumeField<FieldValueType>>
{

public:

    using VectorValueType = FieldValueType;

    // copy constructor
    LaplacianOperator(const LaplacianOperator& lapOp)
        : dsl::OperatorMixin<VolumeField<FieldValueType>>(
            lapOp.exec_, lapOp.coeffs_, lapOp.field_, lapOp.type_
        ),
          gamma_(lapOp.gamma_),
          sameTypeStrategy_(lapOp.sameTypeStrategy_ ? lapOp.sameTypeStrategy_->clone() : nullptr),
          scalarMtxStrategy_(
              lapOp.scalarMtxStrategy_ ? lapOp.scalarMtxStrategy_->clone() : nullptr
          ) {};

    LaplacianOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& gamma,
        VolumeField<FieldValueType>& phi,
        Input input
    )
        : dsl::OperatorMixin<VolumeField<FieldValueType>>(
            phi.exec(), dsl::Coeff(1.0), phi, termType
        ),
          gamma_(gamma),
          sameTypeStrategy_(LaplacianOperatorFactory<FieldValueType, FieldValueType>::create(
              this->exec_, phi.mesh(), input
          )),
          scalarMtxStrategy_(nullptr)
    {
        if constexpr (!std::is_same_v<FieldValueType, scalar>)
        {
            // The first create() consumed tokens; rewind the cursor so the second
            // strategy can read the same scheme tokens from the start.
            if (std::holds_alternative<NeoN::TokenList>(input))
            {
                std::get<NeoN::TokenList>(input).reset();
            }
            scalarMtxStrategy_ = LaplacianOperatorFactory<FieldValueType, scalar>::create(
                this->exec_, phi.mesh(), input
            );
        }
    };

    LaplacianOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& gamma,
        VolumeField<FieldValueType>& phi
    )
        : dsl::OperatorMixin<VolumeField<FieldValueType>>(
            phi.exec(), dsl::Coeff(1.0), phi, termType
        ),
          gamma_(gamma), sameTypeStrategy_(nullptr), scalarMtxStrategy_(nullptr) {};


    void explicitOperation(Vector<FieldValueType>& source) const
    {
        NF_ASSERT(sameTypeStrategy_, "LaplacianOperatorStrategy not initialized");
        const auto operatorScaling = this->getCoefficient();
        NeoN::Vector<FieldValueType> tmpsource(
            source.exec(), source.size(), zero<FieldValueType>()
        );
        sameTypeStrategy_->laplacian(tmpsource, gamma_, this->field_, operatorScaling);
        source += tmpsource;
    }

    void implicitOperation(la::LinearSystem<FieldValueType, FieldValueType>& ls) const
    {
        NF_ASSERT(sameTypeStrategy_, "LaplacianOperatorStrategy not initialized");
        const auto operatorScaling = this->getCoefficient();
        sameTypeStrategy_->laplacian(ls, gamma_, this->field_, operatorScaling);
    }

    /* @brief Implicit assembly into a scalar-matrix / FieldValueType-rhs linear system
     *        (segregated vector-solve form). Only present when FieldValueType != scalar.
     */
    template<typename F = FieldValueType>
        requires(!std::is_same_v<F, scalar>)
    void implicitOperation(la::LinearSystem<scalar, FieldValueType>& ls) const
    {
        NF_ASSERT(scalarMtxStrategy_, "Scalar-matrix LaplacianOperatorStrategy not initialized");
        const auto operatorScaling = this->getCoefficient();
        scalarMtxStrategy_->laplacian(ls, gamma_, this->field_, operatorScaling);
    }

    void read(const Input& input)
    {
        const UnstructuredMesh& mesh = this->field_.mesh();
        NeoN::TokenList tokens;
        if (std::holds_alternative<NeoN::Dictionary>(input))
        {
            auto dict = std::get<NeoN::Dictionary>(input);
            std::string schemeName = "laplacian(" + gamma_.name + "," + this->field_.name + ")";
            tokens = dict.subDict("laplacianSchemes").get<NeoN::TokenList>(schemeName);
        }
        else
        {
            tokens = std::get<NeoN::TokenList>(input);
        }
        sameTypeStrategy_ = LaplacianOperatorFactory<FieldValueType, FieldValueType>::create(
            this->exec(), mesh, tokens
        );
        if constexpr (!std::is_same_v<FieldValueType, scalar>)
        {
            tokens.reset();
            scalarMtxStrategy_ = LaplacianOperatorFactory<FieldValueType, scalar>::create(
                this->exec(), mesh, tokens
            );
        }
    }

    std::string getName() const { return "LaplacianOperator"; }

    // TODO make this private and let only friends use it
    Dictionary getConfig() const
    {
        const auto& ret = this->getVector();
        const auto& coeff = this->getCoefficient();
        return {
            {"field", detail::RefHolder<VolumeField<FieldValueType>> {ret}},
            {"coeff", detail::RefHolder<dsl::Coeff> {coeff}},
            {"gamma", detail::RefHolder<SurfaceField<NeoN::scalar>> {gamma_}}
        };
    }

private:

    const SurfaceField<scalar>& gamma_;

    std::unique_ptr<LaplacianOperatorFactory<FieldValueType, FieldValueType>> sameTypeStrategy_;
    // Only initialized when FieldValueType != scalar; used to assemble into a
    // LinearSystem<scalar, FieldValueType> for segregated vector solves.
    std::unique_ptr<LaplacianOperatorFactory<FieldValueType, scalar>> scalarMtxStrategy_;
};


} // namespace NeoN
