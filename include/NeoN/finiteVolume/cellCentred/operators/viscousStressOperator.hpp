// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// #include <memory>

#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/error.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/dsl/coeff.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred
{

// ----------------------------
// Low-level kernel (decl only)
// ----------------------------

/**
 * @brief Explicit contribution for:
 *   fvc::laplacian(nut, U) - fvc::div(nu * dev2(T(fvc::grad(U))))
 *
 * Adds result into `rhs` (Vector<Vec3>) directly.
 *
 * @note This is called by ViscousStress operators. Not intended to be used from DSL directly.
 */
void computeViscousStressExp(
    const FaceNormalGradient<Vec3>& faceNormalGradient,
    const SurfaceInterpolation<Vec3>& surfaceInterpolationVec,
    const SurfaceField<scalar>& nuF,
    const SurfaceField<scalar>& nutF,
    const VolumeField<Vec3>& U,
    const VolumeField<Vec3>& gradUx,
    const VolumeField<Vec3>& gradUy,
    const VolumeField<Vec3>& gradUz,
    Vector<Vec3>& rhs,
    const dsl::Coeff operatorScaling
);


// ----------------------------
// Operator runtime selection
// ----------------------------

class ViscousStressOperatorFactory :
    public NeoN::RuntimeSelectionFactory<
        ViscousStressOperatorFactory,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{
public:

    static std::unique_ptr<ViscousStressOperatorFactory>
    create(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
    {
        std::string key = (std::holds_alternative<NeoN::Dictionary>(inputs))
                            ? std::get<NeoN::Dictionary>(inputs).get<std::string>("viscousStress")
                            : std::get<NeoN::TokenList>(inputs).next<std::string>();

        ViscousStressOperatorFactory::keyExistsOrError(key);
        return ViscousStressOperatorFactory::table().at(key)(exec, mesh, inputs);
    }

    static std::string name() { return "ViscousStressOperatorFactory"; }

    ViscousStressOperatorFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh)
    {}

    virtual ~ViscousStressOperatorFactory() = default;

    /**
     * @brief Add explicit contribution to rhs of ls.
     *
     * @note Must NOT touch matrix. Explicit operator only.
     */
    virtual void explicitOp(
        Vector<Vec3>& rhs,
        const SurfaceField<scalar>& nuF,
        const SurfaceField<scalar>& nutF,
        const VolumeField<Vec3>& U,
        const VolumeField<Vec3>& gradUx,
        const VolumeField<Vec3>& gradUy,
        const VolumeField<Vec3>& gradUz,
        const dsl::Coeff operatorScaling
    ) const = 0;

    virtual std::unique_ptr<ViscousStressOperatorFactory> clone() const = 0;

protected:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
};


// ----------------------------
// DSL-facing operator wrapper
// ----------------------------

class ViscousStressOperator : public dsl::OperatorMixin<VolumeField<Vec3>>
{
public:

    using VectorValueType = Vec3;

    ViscousStressOperator(const ViscousStressOperator& other)
        : dsl::OperatorMixin<VolumeField<Vec3>>(
            other.exec_, other.coeffs_, other.field_, other.type_
        ),
          nuF_(other.nuF_), nutF_(other.nutF_), gradUx_(other.gradUx_), gradUy_(other.gradUy_),
          gradUz_(other.gradUz_), viscousOp_(other.viscousOp_ ? other.viscousOp_->clone() : nullptr)
    {}

    ViscousStressOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& nuF,
        const SurfaceField<scalar>& nutF,
        const VolumeField<Vec3>& U,
        const VolumeField<Vec3>& gradUx,
        const VolumeField<Vec3>& gradUy,
        const VolumeField<Vec3>& gradUz,
        Input input
    )
        : dsl::OperatorMixin<VolumeField<Vec3>>(U.exec(), dsl::Coeff(1.0), U, termType), nuF_(nuF),
          nutF_(nutF), gradUx_(gradUx), gradUy_(gradUy), gradUz_(gradUz),
          viscousOp_(ViscousStressOperatorFactory::create(this->exec_, U.mesh(), input))
    {}

    ViscousStressOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& nuF,
        const SurfaceField<scalar>& nutF,
        const VolumeField<Vec3>& U,
        const VolumeField<Vec3>& gradUx,
        const VolumeField<Vec3>& gradUy,
        const VolumeField<Vec3>& gradUz,
        std::unique_ptr<ViscousStressOperatorFactory> viscousOp
    )
        : dsl::OperatorMixin<VolumeField<Vec3>>(U.exec(), dsl::Coeff(1.0), U, termType), nuF_(nuF),
          nutF_(nutF), gradUx_(gradUx), gradUy_(gradUy), gradUz_(gradUz),
          viscousOp_(std::move(viscousOp))
    {}

    ViscousStressOperator(
        dsl::Operator::Type termType,
        const SurfaceField<scalar>& nuF,
        const SurfaceField<scalar>& nutF,
        const VolumeField<Vec3>& U,
        const VolumeField<Vec3>& gradUx,
        const VolumeField<Vec3>& gradUy,
        const VolumeField<Vec3>& gradUz
    )
        : dsl::OperatorMixin<VolumeField<Vec3>>(U.exec(), dsl::Coeff(1.0), U, termType), nuF_(nuF),
          nutF_(nutF), gradUx_(gradUx), gradUy_(gradUy), gradUz_(gradUz), viscousOp_(nullptr)
    {}

    void explicitOperation(Vector<Vec3>& source) const
    {
        NF_ASSERT(viscousOp_, "ViscousStressOperatorStrategy not initialized");
        Vector<Vec3> tmpsource(source.exec(), source.size(), zero<Vec3>());
        const auto operatorScaling = this->getCoefficient();
        viscousOp_->explicitOp(
            tmpsource, nuF_, nutF_, this->field_, gradUx_, gradUy_, gradUz_, operatorScaling
        );
        source += tmpsource;
    }

    void implicitOperation(la::LinearSystem<Vec3, localIdx>&) const {}

    void read(const Input& input)
    {
        const UnstructuredMesh& mesh = this->field_.mesh();
        if (std::holds_alternative<NeoN::Dictionary>(input))
        {
            auto dict = std::get<NeoN::Dictionary>(input);
            auto tokens =
                dict.subDict("viscousStressSchemes").get<NeoN::TokenList>("viscousStress");
            viscousOp_ = ViscousStressOperatorFactory::create(this->exec(), mesh, tokens);
        }
        else
        {
            auto tokens = std::get<NeoN::TokenList>(input);
            viscousOp_ = ViscousStressOperatorFactory::create(this->exec(), mesh, tokens);
        }
    }

    std::string getName() const { return "ViscousStressOperator"; }

private:

    const SurfaceField<scalar>& nuF_;
    const SurfaceField<scalar>& nutF_;
    const VolumeField<Vec3>& gradUx_;
    const VolumeField<Vec3>& gradUy_;
    const VolumeField<Vec3>& gradUz_;
    std::unique_ptr<ViscousStressOperatorFactory> viscousOp_;
};


// ----------------------------
// Concrete implementation: Gauss
// ----------------------------

class GaussViscousStress : public ViscousStressOperatorFactory::Register<GaussViscousStress>
{
    using Base = ViscousStressOperatorFactory::Register<GaussViscousStress>;

public:

    static std::string name() { return "Gauss"; }
    static std::string doc()
    {
        return "Gauss explicit viscous stress (tensor-free reconstruction)";
    }
    static std::string schema() { return "none"; }

    GaussViscousStress(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), surfaceInterpolationVec_(exec, mesh, inputs),
          faceNormalGradient_(exec, mesh, inputs)
    {}

    void explicitOp(
        Vector<Vec3>& rhs,
        const SurfaceField<scalar>& nuF,
        const SurfaceField<scalar>& nutF,
        const VolumeField<Vec3>& U,
        const VolumeField<Vec3>& gradUx,
        const VolumeField<Vec3>& gradUy,
        const VolumeField<Vec3>& gradUz,
        const dsl::Coeff operatorScaling
    ) const override;

    VolumeField<Vec3> viscousStress(
        const SurfaceField<scalar>& nuF,
        const SurfaceField<scalar>& nutF,
        const VolumeField<Vec3>& U,
        const VolumeField<Vec3>& gradUx,
        const VolumeField<Vec3>& gradUy,
        const VolumeField<Vec3>& gradUz,
        const dsl::Coeff operatorScaling
    ) const;

    void viscousStress(
        VolumeField<Vec3>& result,
        const SurfaceField<scalar>& nuF,
        const SurfaceField<scalar>& nutF,
        const VolumeField<Vec3>& U,
        const VolumeField<Vec3>& gradUx,
        const VolumeField<Vec3>& gradUy,
        const VolumeField<Vec3>& gradUz,
        const dsl::Coeff operatorScaling
    ) const;

    void viscousStress(
        Vector<Vec3>& result,
        const SurfaceField<scalar>& nuF,
        const SurfaceField<scalar>& nutF,
        const VolumeField<Vec3>& U,
        const VolumeField<Vec3>& gradUx,
        const VolumeField<Vec3>& gradUy,
        const VolumeField<Vec3>& gradUz,
        const dsl::Coeff operatorScaling
    ) const;

    std::unique_ptr<ViscousStressOperatorFactory> clone() const override
    {
        return std::make_unique<GaussViscousStress>(*this);
    }

private:

    SurfaceInterpolation<Vec3> surfaceInterpolationVec_;
    FaceNormalGradient<Vec3> faceNormalGradient_;
};

} // namespace NeoN::finiteVolume::cellCentred
