// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/dsl/spatialOperator.hpp"

// TODO: check if includes necessary
// #include "NeoN/core/dictionary.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN::dsl::temporal
{

template<typename VectorType>
class Ddt : public OperatorMixin<VectorType>
{

public:

    Ddt(VectorType& field)
        : OperatorMixin<VectorType>(field.exec(), field, Operator::Type::Implicit)
    {}

    std::string getName() const { return "TimeOperator"; }

    void explicitOperation(Vector<scalar>&, scalar, scalar) const
    {
        NF_ERROR_EXIT("Not implemented");
    }

    void implicitOperation(la::LinearSystem<scalar, localIdx>&, scalar, scalar) const
    {
        NF_ERROR_EXIT("Not implemented");
    }
};

/* @brief factory function to create a Ddt term as ddt() */
template<typename VectorType>
Ddt<VectorType> ddt(VectorType& in)
{
    return Ddt(in);
};

} // namespace NeoN

namespace NeoN::dsl
{
/* @brief Helper function to access ddtPhiCorr via dsl
 * @param U The velocity volume field
 * @param phi The flux surface field
 * @param dt The time step size
 * @param fvSchemes The fvSchemes dictionary
 * @parame fvSolution The fvSolution dictionary
 */
inline fvcc::SurfaceField<scalar> ddtPhiCorr(
    const fvcc::VolumeField<Vec3>& U,
    const fvcc::SurfaceField<scalar>& phi,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution
)
{
    using VolVector = fvcc::VolumeField<Vec3>;
    using TimeInt = NeoN::timeIntegration::TimeIntegration<VolVector>;

    // Mirror the scheme selection used in solver.hpp:
    // TimeIntegration is built from ddtSchemes + fvSolution.
    const Dictionary& ddtSchemes = fvSchemes.subDict("ddtSchemes");

    TimeInt integrator(ddtSchemes, fvSolution);

    // Delegate to scheme-specific implementation in TimeIntegratorBase subclasses
    return integrator.ddtPhiCorr(U, phi, dt);
}

} // namespace NeoN::dsl
