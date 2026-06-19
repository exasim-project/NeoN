// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>

#include "NeoN/core/error.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gradOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @class CellLimitedGrad
 * @brief Slope-limited cell gradient.
 *
 * Wraps a runtime-selected base gradient scheme and clips its reconstructed
 * gradient so that values extrapolated from the cell centre to the surrounding
 * faces stay bounded by the minimum and maximum of the cell's own value and its
 * face-neighbour cell values. Uses the minmod limiter, limiter(r) = min(r, 1),
 * applied identically to every gradient component.
 *
 * Scheme specification (gradSchemes / TokenList):
 *   cellLimited <baseScheme ...> <k>
 * e.g.  cellLimited Gauss linear 1
 * where 0 <= k <= 1 sets the limiting strength: k = 0 disables limiting (returns
 * the unlimited base gradient), k = 1 is the strongest, most stable limiting,
 * and intermediate values widen the admissible bounds accordingly.
 *
 * v1 assembles the explicit gradient on cell-centred fields; the implicit
 * (LinearSystem) path is not provided. The kernels use face-scatter atomics so
 * the design extends to device and distributed (processor-boundary) execution;
 * only CPU-serial is validated in v1.
 */
class CellLimitedGrad : public GradOperatorFactory<Vec3>::template Register<CellLimitedGrad>
{
    using Base = GradOperatorFactory<Vec3>::template Register<CellLimitedGrad>;

    // Limiter strength used when no coefficient is supplied (strongest limiting).
    static constexpr scalar DEFAULT_COEFF = scalar(1);

public:

    static std::string name() { return "cellLimited"; }

    static std::string doc() { return "Cell limited gradient (minmod slope limiter)"; }

    static std::string schema() { return "none"; }

    CellLimitedGrad(const Executor& exec, const UnstructuredMesh& mesh, const Input& inputs);

    /* @brief implicit gradient assembly — not supported for the cell-limited scheme */
    virtual void
    grad(const VolumeField<scalar>&, const dsl::Coeff, la::LinearSystem<Vec3>&) const override
    {
        NF_ERROR_EXIT("Not implemented");
    };

    /* @brief explicit limited gradient assembled into the supplied vector */
    virtual void grad(
        const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling, Vector<Vec3>& gradPhi
    ) const override;

    /* @brief explicit limited gradient returned as a new VolumeField */
    VolumeField<Vec3> grad(
        const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling = dsl::Coeff {}
    ) const override;

    virtual std::unique_ptr<GradOperatorFactory<Vec3>> clone() const override
    {
        NF_ERROR_EXIT("Not implemented");
        return nullptr;
    };

private:

    /* @brief read the limiter coefficient k from the scheme Input */
    static scalar readCoeff(const Input& inputs);

    // Declared before coeff_ so the base scheme consumes its tokens from the
    // shared Input cursor before readCoeff reads the trailing coefficient.
    std::unique_ptr<GradOperatorFactory<Vec3>> baseGradScheme_;
    scalar coeff_;
    // Supplies the cached internal-face cell-to-face displacement vectors; the raw
    // mesh centres are freed once the geometry scheme is built.
    std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoN::finiteVolume::cellCentred
