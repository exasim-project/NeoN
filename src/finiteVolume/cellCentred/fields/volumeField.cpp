// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/core/macros.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{


template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string name,
    const UnstructuredMesh& mesh,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
)
    : DomainMixin<ValueType>(
        exec, name, mesh, Field<ValueType>(exec, mesh.nCells(), mesh.boundaryMesh().offset())
    ),
      key(""), fieldCollectionName(""), boundaryConditions_(boundaryConditions), db_(std::nullopt)
{}

template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string name,
    const UnstructuredMesh& mesh,
    const Vector<ValueType>& internalVector,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
)
    : DomainMixin<ValueType>(
        exec, name, mesh, Field<ValueType>(exec, internalVector, mesh.boundaryMesh().offset())
    ),
      key(""), fieldCollectionName(""), boundaryConditions_(boundaryConditions), db_(std::nullopt)
{}

template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string name,
    const UnstructuredMesh& mesh,
    const Vector<ValueType>& internalVector,
    const BoundaryData<ValueType>& boundaryVectors,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
)
    : DomainMixin<ValueType>(exec, name, mesh, internalVector, boundaryVectors), key(""),
      fieldCollectionName(""), boundaryConditions_(boundaryConditions), db_(std::nullopt)
{}


template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string fieldName,
    const UnstructuredMesh& mesh,
    const Field<ValueType>& domainVector,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions,
    Database& db,
    std::string dbKey,
    std::string collectionName
)
    : DomainMixin<ValueType>(exec, fieldName, mesh, domainVector), key(dbKey),
      fieldCollectionName(collectionName), boundaryConditions_(boundaryConditions), db_(&db)
{}

template<typename ValueType>
VolumeField<ValueType>::VolumeField(const VolumeField& other)
    : DomainMixin<ValueType>(other), key(other.key), fieldCollectionName(other.fieldCollectionName),
      boundaryConditions_(other.boundaryConditions_), db_(other.db_)
{}

template<typename ValueType>
VolumeField<ValueType>& VolumeField<ValueType>::operator+=(const ValueType rhs)
{
    add(this->internalVector(), rhs);
    correctBoundaryConditions();
    return *this;
}

template<typename ValueType>
VolumeField<ValueType>& VolumeField<ValueType>::operator-=(const ValueType rhs)
{
    sub(this->internalVector(), rhs);
    correctBoundaryConditions();
    return *this;
}

template<typename ValueType>
void VolumeField<ValueType>::correctBoundaryConditions()
{
    for (auto& boundaryCondition : boundaryConditions_)
    {
        boundaryCondition.correctBoundaryCondition(this->field_);
    }
}

#define NN_DECLARE_FIELD(TYPENAME) template class VolumeField<TYPENAME>

NN_FOR_ALL_VALUE_TYPES(NN_DECLARE_FIELD);

}
