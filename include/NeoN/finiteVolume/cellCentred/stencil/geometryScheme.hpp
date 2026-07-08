// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once


#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred
{

class GeometrySchemeFactory
{

public:

    GeometrySchemeFactory();

    virtual ~GeometrySchemeFactory() = default;

    virtual void updateWeights(const Executor& exec, SurfaceField<scalar>& weights) = 0;

    virtual void
    updateNonOrthDeltaCoeffs(const Executor& exec, SurfaceField<scalar>& nonOrthDeltaCoeffs) = 0;

    // nonOrthDeltaCoeffs is the precomputed 1/(n.d) field (must be updated first); the
    // correction vectors read it rather than re-deriving the formula, keeping a single
    // source of truth for the coefficient.
    virtual void updateNonOrthCorrectionVec3s(
        const Executor& exec,
        const SurfaceField<scalar>& nonOrthDeltaCoeffs,
        SurfaceField<Vec3>& nonOrthCorrectionVec3s
    ) = 0;
};

/* @class GeometryScheme
 * @brief Implements access to compute weights and nonOrthDeltaCoeffs
 *
 * Where:
 *  - weight: the distance of the cell centre to face normalized by the distance to the neighbour
 * cell
 *  - nonOrthDeltaCoeff: 1 / (faceNormal . cellToCellDist), the over-relaxed (non-orthogonal)
 * inverse distance, floored at 1 / (0.05 * |cellToCellDist|)
 */
class GeometryScheme
{
public:

    GeometryScheme(
        const Executor& exec,
        std::unique_ptr<GeometrySchemeFactory> kernel,
        const SurfaceField<scalar>& weights,
        const SurfaceField<scalar>& nonOrthDeltaCoeffs,
        const SurfaceField<Vec3>& nonOrthCorrectionVec3s
    );

    GeometryScheme(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::unique_ptr<GeometrySchemeFactory> kernel
    );

    GeometryScheme(const UnstructuredMesh& mesh // will lookup the kernel
    );

    virtual ~GeometryScheme() = default;

    const SurfaceField<scalar>& weights() const;

    const SurfaceField<scalar>& nonOrthDeltaCoeffs() const;

    const SurfaceField<Vec3>& nonOrthCorrectionVec3s() const;

    // Vector from the owner cell centre to the face centre (Cf - C_own), one per internal face.
    // Cached here because the underlying cell/face centres are freed by reset(); schemes that need
    // a cell-to-face offset (e.g. linearUpwind's gradient correction) read it from this object.
    const SurfaceField<Vec3>& faceDeltaOwner() const;

    // Vector from the neighbour cell centre to the face centre (Cf - C_nei), one per internal face.
    const SurfaceField<Vec3>& faceDeltaNeighbour() const;

    void update();

    // Frees the mesh's per-cell/face centre arrays after update() to save device memory;
    // they are not needed once the geometry-scheme fields are cached.
    // TODO: check if we can remove the temporary fields from the unstructured mesh
    // altogether: compute the geometry-scheme data explicitly first and pass it as an
    // argument, instead of freeing mesh members after the fact.
    void reset() const;

    std::string name() const;

    // add selection mechanism via dictionary later
    static const std::shared_ptr<GeometryScheme> readOrCreate(const UnstructuredMesh& mesh);

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    std::unique_ptr<GeometrySchemeFactory> kernel_;

    SurfaceField<scalar> weights_;
    SurfaceField<scalar> nonOrthDeltaCoeffs_;
    SurfaceField<Vec3> nonOrthCorrectionVec3s_;
    SurfaceField<Vec3> faceDeltaOwner_;
    SurfaceField<Vec3> faceDeltaNeighbour_;
};

} // namespace NeoN
