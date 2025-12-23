// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"
#include <string>

#include "../dsl/common.hpp"

#include "NeoN/NeoN.hpp"


namespace fvcc = NeoN::finiteVolume::cellCentred;

using Vector = NeoN::Vector<NeoN::scalar>;
using Coeff = NeoN::dsl::Coeff;
using SpatialOperator = NeoN::dsl::SpatialOperator<NeoN::scalar>;
using TemporalOperator = NeoN::dsl::TemporalOperator<NeoN::scalar>;
using Executor = NeoN::Executor;
using VolumeField = fvcc::VolumeField<NeoN::scalar>;
using OperatorMixin = NeoN::dsl::OperatorMixin<VolumeField>;
using BoundaryData = NeoN::BoundaryData<NeoN::scalar>;

// only for msvc
template class NeoN::timeIntegration::RungeKutta<VolumeField>;

class YSquared : public OperatorMixin
{

public:

    using VectorValueType = NeoN::scalar;

    YSquared(VolumeField& field)
        : OperatorMixin(field.exec(), dsl::Coeff(1.0), field, Operator::Type::Explicit)
    {}

    void explicitOperation(Vector& source) const
    {
        auto sourceView = source.view();
        auto fieldData = field_.internalVector().data();
        NeoN::parallelFor(
            source.exec(),
            source.range(),
            KOKKOS_LAMBDA(const localIdx i) { sourceView[i] -= fieldData[i] * fieldData[i]; }
        );
    }

    std::string getName() const { return "YSquared"; }
};

TEST_CASE("TimeIntegration - Runge Kutta")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    NeoN::scalar convergenceTolerance = 1.0e-4; // how much lower we accept that expected order.

    // Set up dictionary.
    NeoN::Database db;
    NeoN::Dictionary fvSchemes;
    NeoN::Dictionary timeIntegrationDict;
    // ddtSchemes.insert("default", std::string("Euler"));
    timeIntegrationDict.insert("type", std::string("Runge-Kutta"));
    timeIntegrationDict.insert("Runge-Kutta-Method", std::string("Forward-Euler"));
    fvSchemes.insert("timeIntegration", timeIntegrationDict);
    NeoN::Dictionary fvSolution;

    // Set up fields.
    auto mesh = NeoN::createSingleCellMesh(exec);
    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "fieldCollection");
    fvcc::VolumeField<NeoN::scalar>& vf =
        fieldCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
            CreateVector {.name = "vf", .mesh = mesh, .timeIndex = 1}
        );

    // Setup solve parameters.
    const NeoN::scalar maxTime = 0.1;
    const NeoN::scalar initialValue = 1.0;
    std::array<NeoN::scalar, 2> deltaTime = {0.01, 0.001};

    SECTION("Solve on " + execName)
    {
        std::size_t iTest = 0;
        std::array<NeoN::scalar, 2> error;
        for (auto dt : deltaTime)
        {
            // reset
            auto& vfOld = fvcc::oldTime(vf);
            NeoN::scalar time = 0.0;
            vf.internalVector() = initialValue;
            vfOld.internalVector() = initialValue;

            // Set expression
            TemporalOperator ddtOp = NeoN::dsl::ddt(vfOld);

            // Build ODE:
            //   dU/dt + U^2 = 0
            //   dU/dt = -U^2
            //
            // IMPORTANT:
            // - ddt(vf): time derivative of the evolving state
            // - YSquared(vf): RHS evaluated at current RK stage
            //
            auto divOp = YSquared(vf);
            auto eqn = ddtOp + divOp;

            // solve.
            while (time < maxTime)
            {
                NeoN::dsl::solve(eqn, vf, time, dt, fvSchemes, fvSolution);
                time += dt;
            }

            // Analytical solution:
            //   dU/dt = -U^2
            //   U(t) = 1 / (U0^{-1} + t)
            // check error.
            NeoN::scalar analytical = 1.0 / (initialValue - maxTime);
            auto vfHost = vf.internalVector().copyToHost();
            error[iTest] = std::abs(vfHost.view()[0] - analytical);
            iTest++;
        }

        // check order of convergence.
        NeoN::scalar order = (std::log(error[0]) - std::log(error[1]))
                           / (std::log(deltaTime[0]) - std::log(deltaTime[1]));
        REQUIRE(order > (1.0 - convergenceTolerance));
    }
}
