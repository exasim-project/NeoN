// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/vector/vectorFreeFunctions.hpp"
#include "NeoN/core/macros.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{


template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string nameIn,
    const UnstructuredMesh& mesh,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
)
    : DomainMixin<ValueType>(
        exec, nameIn, mesh, Field<ValueType>(exec, mesh.nCells(), mesh.boundaryMesh().offset())
    ),
      FieldDatabaseMixin(), boundaryConditions_(boundaryConditions)
{}

template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string nameIn,
    const UnstructuredMesh& mesh,
    const Vector<ValueType>& internalVector,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
)
    : DomainMixin<ValueType>(
        exec, nameIn, mesh, Field<ValueType>(exec, internalVector, mesh.boundaryMesh().offset())
    ),
      FieldDatabaseMixin(), boundaryConditions_(boundaryConditions)
{}

template<typename ValueType>
VolumeField<ValueType>::VolumeField(
    const Executor& exec,
    std::string nameIn,
    const UnstructuredMesh& mesh,
    const Vector<ValueType>& internalVector,
    const BoundaryData<ValueType>& boundaryVectors,
    const std::vector<VolumeBoundary<ValueType>>& boundaryConditions
)
    : DomainMixin<ValueType>(exec, nameIn, mesh, internalVector, boundaryVectors),
      FieldDatabaseMixin(), boundaryConditions_(boundaryConditions)
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
    : DomainMixin<ValueType>(exec, fieldName, mesh, domainVector),
      FieldDatabaseMixin(db, std::move(dbKey), std::move(collectionName)),
      boundaryConditions_(boundaryConditions)
{}

template<typename ValueType>
VolumeField<ValueType>::VolumeField(const VolumeField& other)
    : DomainMixin<ValueType>(other), FieldDatabaseMixin(other),
      boundaryConditions_(other.boundaryConditions_)
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
    // One-time set() pass: initialise patch data that is constant for the lifetime of the field
    // (e.g. the processor BC's unused mixed-BC coefficients). Runs once per field instance.
    if (!boundaryConditionsSet_)
    {
        for (auto& boundaryCondition : boundaryConditions_)
        {
            boundaryCondition.set(this->field_);
        }
        boundaryConditionsSet_ = true;
    }

    // Per-iteration update(): each processor BC seeds its owner value and posts a non-blocking
    // isend/irecv WITHOUT draining (valueNoWait), so that ALL proc patches are in flight before
    // any completes — completing patch-by-patch mid-loop serialises the exchange and drops the
    // second patch's halo on a rank that owns more than one.
    for (auto& boundaryCondition : boundaryConditions_)
    {
        boundaryCondition.update(this->field_);
    }
    // Drain the processor-halo exchange once, after all proc patches have posted. This leaves
    // boundaryData().value() holding the true neighbour values on return; without it the result
    // would depend on whether a later value() read happens to drain first (it often does not),
    // silently leaving the owner seed at processor faces.
    this->field_.boundaryData().waitAll();
}

template<typename ValueType>
void VolumeField<ValueType>::correctBoundaryConditions(const BoundaryContext& ctx)
{
    for (auto& boundaryCondition : boundaryConditions_)
    {
        boundaryCondition.correctBoundaryCondition(this->field_, ctx);
    }
    this->field_.boundaryData().waitAll();
}

#define NN_DECLARE_FIELD(TYPENAME) template class VolumeField<TYPENAME>

NN_FOR_ALL_VALUE_TYPES(NN_DECLARE_FIELD);

}
