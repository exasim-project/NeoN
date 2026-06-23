// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once


#include <optional>

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
    // Computed lazily on first access (or via ensureFaceDeltas()); only schemes that need a
    // cell-to-face offset (e.g. linearUpwind's gradient correction) ever trigger the allocation,
    // so a run that never selects such a scheme pays nothing.
    const SurfaceField<Vec3>& faceDeltaOwner() const;

    // Vector from the neighbour cell centre to the face centre (Cf - C_nei), one per internal face.
    const SurfaceField<Vec3>& faceDeltaNeighbour() const;

    // Opt-in trigger for the faceDelta* fields. Computes them (from the still-live mesh centres)
    // and then releases the source geometry. Consumers that need faceDelta* (linearUpwind) call
    // this in their constructor — while the mesh centres are guaranteed alive — because reset()
    // frees those centres on the first read of any cached geometry field. Idempotent; const so it
    // is reachable through the shared (read-only) GeometryScheme handle. No-op cost after the first
    // call is a single bool check.
    void ensureFaceDeltas() const;

    void update();

    // Frees the mesh's per-cell/face centre arrays once the geometry-scheme fields are cached, to
    // save device memory. Deferred (lazy + idempotent): triggered on the first read of any cached
    // geometry field, or by ensureFaceDeltas() right after the faceDelta* fields are built — not in
    // update(), so that a late faceDelta* opt-in can still read the centres.
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

    // Lazily allocated and filled: empty until a consumer opts in via ensureFaceDeltas() (or first
    // reads them). A run that never selects linearUpwind leaves these nullopt, saving
    // 2 * nInternalFaces * sizeof(Vec3) of device memory (~2.6 GB on an 18M-cell mesh). mutable so
    // the const accessors / ensureFaceDeltas() can populate them through the shared handle.
    mutable std::optional<SurfaceField<Vec3>> faceDeltaOwner_;
    mutable std::optional<SurfaceField<Vec3>> faceDeltaNeighbour_;
    mutable bool faceDeltasComputed_ = false;
    // Guards the one-shot, deferred release of the mesh centre arrays (see reset()).
    mutable bool sourceGeometryReleased_ = false;
};

} // namespace NeoN
