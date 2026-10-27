// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/primitives/tensor.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/// BoundaryContext value that a solver sets (to a positive number) for an evaluation that follows
/// a solve. Dirichlet patches that OpenFOAM refreshes only in updateCoeffs(), once when the matrix
/// of the field is built (fixedMean, plenumPressure), then keep their values.
inline constexpr const char* evaluateOnlyKey = "evaluateOnly";

template<typename ValueType>
class VolumeField;

template<typename ValueType>
class SurfaceField;

/**
 * @class BoundaryContext
 * @brief Carries named field references for context-aware boundary conditions.
 *
 * Boundary conditions that need auxiliary fields (U, nu, nearWallDist, omega, k …)
 * look up what they need by name instead of accepting a fixed argument list.
 * Volume fields cover wall functions; the surface-scalar slot carries the face
 * flux phi consumed by flux-dependent BCs such as inletOutlet. Plain scalar values
 * (not fields) cover per-call solver state a BC needs but that has no field of its
 * own, such as the current simulation time or timestep deltaT.
 * All fields are stored as non-owning const pointers; the caller is responsible
 * for ensuring the lifetime of the referenced fields exceeds the context.
 */
class BoundaryContext
{
public:

    void insert(std::string name, const VolumeField<scalar>& f);
    void insert(std::string name, const VolumeField<Vec3>& f);
    void insert(std::string name, const VolumeField<Tensor>& f);
    void insert(std::string name, const SurfaceField<scalar>& f);
    void insertValue(std::string name, scalar value);

    const VolumeField<scalar>& scalarFieldPtr(const std::string& name) const;
    const VolumeField<Vec3>& vectorFieldPtr(const std::string& name) const;
    const VolumeField<Tensor>& tensorFieldPtr(const std::string& name) const;
    const SurfaceField<scalar>& surfaceScalarField(const std::string& name) const;
    scalar value(const std::string& name) const;

    bool hasScalar(const std::string& name) const noexcept;
    bool hasVector(const std::string& name) const noexcept;
    bool hasTensor(const std::string& name) const noexcept;
    bool hasSurfaceScalar(const std::string& name) const noexcept;
    bool hasValue(const std::string& name) const noexcept;

private:

    std::unordered_map<std::string, const VolumeField<NeoN::scalar>*> scalarFields_;
    std::unordered_map<std::string, const VolumeField<Vec3>*> vectorFields_;
    std::unordered_map<std::string, const VolumeField<Tensor>*> tensorFields_;
    std::unordered_map<std::string, const SurfaceField<NeoN::scalar>*> surfaceScalarFields_;
    std::unordered_map<std::string, scalar> values_;
};

} // namespace NeoN::finiteVolume::cellCentred
