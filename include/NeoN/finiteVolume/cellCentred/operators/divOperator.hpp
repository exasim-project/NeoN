// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
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
template<typename FieldValueType, typename AssemblyType = FieldValueType>
class DivOperatorFactory :
    public RuntimeSelectionFactory<
        DivOperatorFactory<FieldValueType, AssemblyType>,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{

public:

    static std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>>
    create(const Executor& exec, const UnstructuredMesh& uMesh, const Input& inputs)
    {
        std::string key = (std::holds_alternative<Dictionary>(inputs))
                            ? std::get<Dictionary>(inputs).get<std::string>("DivOperator")
                            : std::get<TokenList>(inputs).next<std::string>();
        DivOperatorFactory<FieldValueType, AssemblyType>::keyExistsOrError(key);
        return DivOperatorFactory<FieldValueType, AssemblyType>::table().at(key)(
            exec, uMesh, inputs
        );
    }

    static std::string name() { return "DivOperatorFactory"; }

    DivOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~DivOperatorFactory() {} // Virtual destructor

    virtual void
    div(VolumeField<FieldValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const = 0;

    virtual void
    div(la::LinearSystem<AssemblyType, FieldValueType>& ls,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const = 0;

    virtual void
    div(Vector<FieldValueType>& divPhi,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const = 0;

    virtual VolumeField<FieldValueType>
    div(const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        const dsl::Coeff operatorScaling) const = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<DivOperatorFactory<FieldValueType, AssemblyType>> clone() const = 0;

protected:

    const Executor exec_;

    const UnstructuredMesh& mesh_;
};

template<typename FieldValueType>
class DivOperator : public dsl::OperatorMixin<VolumeField<FieldValueType>>
{

public:

    using VectorValueType = FieldValueType;

    // copy constructor
    DivOperator(const DivOperator& divOp)
        : dsl::OperatorMixin<VolumeField<FieldValueType>>(
            divOp.exec_, divOp.coeffs_, divOp.field_, divOp.type_
        ),
          faceFlux_(divOp.faceFlux_),
          sameTypeStrategy_(divOp.sameTypeStrategy_ ? divOp.sameTypeStrategy_->clone() : nullptr),
          scalarMtxStrategy_(
              divOp.scalarMtxStrategy_ ? divOp.scalarMtxStrategy_->clone() : nullptr
          ) {};

    DivOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi,
        Input input
    )
        : dsl::OperatorMixin<VolumeField<FieldValueType>>(
            phi.exec(), dsl::Coeff(1.0), phi, termType
        ),
          faceFlux_(faceFlux),
          sameTypeStrategy_(DivOperatorFactory<FieldValueType, FieldValueType>::create(
              phi.exec(), phi.mesh(), input
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
            scalarMtxStrategy_ =
                DivOperatorFactory<FieldValueType, scalar>::create(phi.exec(), phi.mesh(), input);
        }
    };

    DivOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<FieldValueType>& phi
    )
        : dsl::OperatorMixin<VolumeField<FieldValueType>>(
            phi.exec(), dsl::Coeff(1.0), phi, termType
        ),
          faceFlux_(faceFlux), sameTypeStrategy_(nullptr), scalarMtxStrategy_(nullptr) {};


    void explicitOperation(Vector<FieldValueType>& source) const
    {
        NF_ASSERT(sameTypeStrategy_, "DivOperatorStrategy not initialized");
        auto tmpsource =
            Vector<FieldValueType>(source.exec(), source.size(), zero<FieldValueType>());
        const auto operatorScaling = this->getCoefficient();
        sameTypeStrategy_->div(tmpsource, faceFlux_, this->getVector(), operatorScaling);
        source += tmpsource;
    }

    void implicitOperation(la::LinearSystem<FieldValueType, FieldValueType>& ls) const
    {
        NF_ASSERT(sameTypeStrategy_, "DivOperatorStrategy not initialized");
        const auto operatorScaling = this->getCoefficient();
        sameTypeStrategy_->div(ls, faceFlux_, this->getVector(), operatorScaling);
    }

    /* @brief Implicit assembly into a scalar-matrix / FieldValueType-rhs linear system
     *        (segregated vector-solve form). Only present when FieldValueType != scalar;
     *        for scalar fields the same-type overload above already covers this signature.
     */
    template<typename F = FieldValueType>
        requires(!std::is_same_v<F, scalar>)
    void implicitOperation(la::LinearSystem<scalar, FieldValueType>& ls) const
    {
        NF_ASSERT(scalarMtxStrategy_, "Scalar-matrix DivOperatorStrategy not initialized");
        const auto operatorScaling = this->getCoefficient();
        scalarMtxStrategy_->div(ls, faceFlux_, this->getVector(), operatorScaling);
    }

    void read(const Input& input)
    {
        const UnstructuredMesh& mesh = this->getVector().mesh();
        NeoN::TokenList tokens;
        if (std::holds_alternative<NeoN::Dictionary>(input))
        {
            auto dict = std::get<NeoN::Dictionary>(input);
            std::string schemeName = "div(" + faceFlux_.name + "," + this->getVector().name + ")";
            tokens = dict.subDict("divSchemes").get<NeoN::TokenList>(schemeName);
        }
        else
        {
            tokens = std::get<NeoN::TokenList>(input);
        }
        sameTypeStrategy_ =
            DivOperatorFactory<FieldValueType, FieldValueType>::create(this->exec(), mesh, tokens);
        if constexpr (!std::is_same_v<FieldValueType, scalar>)
        {
            tokens.reset();
            scalarMtxStrategy_ =
                DivOperatorFactory<FieldValueType, scalar>::create(this->exec(), mesh, tokens);
        }
    }

    std::string getName() const { return "DivOperator"; }

    // TODO make this private and let only friends use it
    Dictionary getConfig() const
    {
        const auto& ret = this->getVector();
        const auto& coeff = this->getCoefficient();
        return {
            {"field", detail::RefHolder<VolumeField<FieldValueType>> {ret}},
            {"coeff", detail::RefHolder<dsl::Coeff> {coeff}},
            {"flux", detail::RefHolder<SurfaceField<NeoN::scalar>> {faceFlux_}}
        };
    }

private:

    const SurfaceField<NeoN::scalar>& faceFlux_;

    std::unique_ptr<DivOperatorFactory<FieldValueType, FieldValueType>> sameTypeStrategy_;
    // Only initialized when FieldValueType != scalar; used to assemble into a
    // LinearSystem<scalar, FieldValueType> for segregated vector solves.
    std::unique_ptr<DivOperatorFactory<FieldValueType, scalar>> scalarMtxStrategy_;
};


} // namespace NeoN
