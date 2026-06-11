// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/error.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLimitedCorrectedFaceNormalGrad(
    const VolumeField<ValueType>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    scalar limitCoeff,
    SurfaceField<ValueType>& surfaceField
);

template<typename ValueType>
void computeLimitedCorrectionTerm(
    const VolumeField<ValueType>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    scalar limitCoeff,
    SurfaceField<ValueType>& corrField
);

template<typename ValueType>
class LimitedCorrected :
    public FaceNormalGradientFactory<ValueType>::template Register<LimitedCorrected<ValueType>>
{
    using Base =
        FaceNormalGradientFactory<ValueType>::template Register<LimitedCorrected<ValueType>>;

    static constexpr scalar DEFAULT_LIMIT_COEFF = 0.333;

public:

    LimitedCorrected(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)),
          limitCoeff_(readLimitCoeff(inputs)) {};

    LimitedCorrected(const Executor& exec, const UnstructuredMesh& mesh)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)),
          limitCoeff_(DEFAULT_LIMIT_COEFF) {};

    static std::string name() { return "limited"; }

    static std::string doc()
    {
        return "Limited corrected face normal gradient with bounded non-orthogonality correction";
    }

    static std::string schema() { return "none"; }

    virtual void faceNormalGrad(
        const VolumeField<ValueType>& volField, SurfaceField<ValueType>& surfaceField
    ) const override
    {
        computeLimitedCorrectedFaceNormalGrad(volField, geometryScheme_, limitCoeff_, surfaceField);
    }

    virtual const SurfaceField<scalar>& deltaCoeffs() const override
    {
        return geometryScheme_->nonOrthDeltaCoeffs();
    }

    bool hasImplicitCorrection() const override { return true; }

    virtual void implicitCorrection(
        const VolumeField<ValueType>& phi, SurfaceField<ValueType>& corrField
    ) const override
    {
        computeLimitedCorrectionTerm(phi, geometryScheme_, limitCoeff_, corrField);
    }

    std::unique_ptr<FaceNormalGradientFactory<ValueType>> clone() const override
    {
        return std::make_unique<LimitedCorrected>(*this);
    }

private:

    static scalar readLimitCoeff(const Input& inputs)
    {
        if (std::holds_alternative<NeoN::Dictionary>(inputs))
        {
            const auto& dict = std::get<NeoN::Dictionary>(inputs);
            if (dict.contains("limitCoeff"))
            {
                return dict.get<scalar>("limitCoeff");
            }
            return DEFAULT_LIMIT_COEFF;
        }
        // TokenList: accept either OF form
        //   terse:   "limited" <coeff>
        //   verbose: "limited" "corrected" <coeff>
        // The factory has already consumed "limited"; consume an optional
        // "corrected" sub-scheme token if present, then read the coefficient.
        const auto& tl = std::get<NeoN::TokenList>(inputs);
        if (tl.peekIs<std::string>())
        {
            const std::string& sub = tl.next<std::string>();
            NF_ASSERT(
                sub == "corrected", "limited snGrad: only 'corrected' sub-scheme is supported"
            );
        }
        return tl.next<scalar>();
    }

    const std::shared_ptr<GeometryScheme> geometryScheme_;
    scalar limitCoeff_;
};

// instantiate the template class
template class LimitedCorrected<scalar>;
template class LimitedCorrected<Vec3>;

} // namespace NeoN
