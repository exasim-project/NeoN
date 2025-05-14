// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void surfaceIntegrate(
    const Executor& exec,
    localIdx nInternalFaces,
    View<const int> neighbour,
    View<const int> owner,
    View<const int> faceCells,
    View<const ValueType> flux,
    View<const scalar> v,
    View<ValueType> res,
    const dsl::Coeff operatorScaling
);

template<typename ValueType>
class SurfaceIntegrate
{

public:

    using VectorValueType = ValueType;

    SurfaceIntegrate(const SurfaceField<ValueType>& flux)
        : flux_(flux), type_(dsl::Operator::Type::Explicit), coeffs_(1.0) {};

    SurfaceIntegrate(const SurfaceIntegrate& surfaceIntegrate)
        : flux_(surfaceIntegrate.flux_), type_(surfaceIntegrate.type_),
          coeffs_(surfaceIntegrate.coeffs_) {};


    void read(const Input&) {}

    void explicitOperation(Vector<ValueType>& source) const
    {
        NeoN::Vector<ValueType> tmpsource(source.exec(), source.size(), zero<ValueType>());
        const auto operatorScaling = this->getCoefficient();

        const UnstructuredMesh& mesh = flux_.mesh();
        const auto exec = flux_.exec();

        auto nInternalFaces = mesh.nInternalFaces();
        surfaceIntegrate<ValueType>(
            exec,
            nInternalFaces,
            mesh.faceNeighbour().view(),
            mesh.faceOwner().view(),
            mesh.boundaryMesh().faceCells().view(),
            this->flux_.internalVector().view(),
            mesh.cellVolumes().view(),
            tmpsource.view(),
            operatorScaling
        );
        source += tmpsource;
    }

    const Executor& exec() const { return flux_.exec(); }

    dsl::Coeff& getCoefficient() { return coeffs_; }

    const dsl::Coeff& getCoefficient() const { return coeffs_; }

    dsl::Operator::Type getType() const { return type_; }

    std::string getName() const { return "SurfaceIntegrate"; }

private:

    const SurfaceField<ValueType>& flux_;
    dsl::Operator::Type type_;
    dsl::Coeff coeffs_;
};


} // namespace NeoN
