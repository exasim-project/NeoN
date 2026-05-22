// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

void BoundaryContext::insert(std::string name, const VolumeField<NeoN::scalar>& f)
{
    scalarFields_.emplace(std::move(name), &f);
}

void BoundaryContext::insert(std::string name, const VolumeField<Vec3>& f)
{
    vectorFields_.emplace(std::move(name), &f);
}

void BoundaryContext::insert(std::string name, const VolumeField<Tensor>& f)
{
    tensorFields_.emplace(std::move(name), &f);
}

const VolumeField<NeoN::scalar>& BoundaryContext::scalarFieldPtr(const std::string& name) const
{
    return *scalarFields_.at(name);
}

const VolumeField<Vec3>& BoundaryContext::vectorFieldPtr(const std::string& name) const
{
    return *vectorFields_.at(name);
}

const VolumeField<Tensor>& BoundaryContext::tensorFieldPtr(const std::string& name) const
{
    return *tensorFields_.at(name);
}

bool BoundaryContext::hasScalar(const std::string& name) const noexcept
{
    return scalarFields_.count(name) > 0;
}

bool BoundaryContext::hasVector(const std::string& name) const noexcept
{
    return vectorFields_.count(name) > 0;
}

bool BoundaryContext::hasTensor(const std::string& name) const noexcept
{
    return tensorFields_.count(name) > 0;
}

} // namespace NeoN::finiteVolume::cellCentred
