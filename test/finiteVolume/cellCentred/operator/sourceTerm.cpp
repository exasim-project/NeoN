// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

using Operator = NeoN::dsl::Operator;

namespace NeoN
{

TEMPLATE_TEST_CASE("SourceTerm", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = createSingleCellMesh(exec);
    auto sp = NeoN::finiteVolume::cellCentred::SparsityPattern {mesh};

    auto coeffBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> coeff(exec, "coeff", mesh, coeffBCs);
    fill(coeff.internalVector(), 2.0);
    fill(coeff.boundaryVector().value(), 0.0);
    coeff.correctBoundaryConditions();

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    fill(phi.internalVector(), 10 * one<TestType>());
    fill(phi.boundaryVector().value(), zero<TestType>());
    phi.correctBoundaryConditions();


    SECTION("explicit SourceTerm" + execName)
    {
        fvcc::SourceTerm<TestType> sTerm(Operator::Type::Explicit, coeff, phi);

        auto source = Vector<TestType>(exec, phi.size(), zero<TestType>());
        sTerm.explicitOperation(source);

        // mesh has one cell
        auto hostSource = source.copyToHost();
        auto hostSourceView = hostSource.view();
        for (auto ii = 0; ii < hostSource.size(); ++ii)
        {
            REQUIRE(hostSourceView[ii] - 20 * one<TestType>() == TestType(0.0));
        }
    }

    SECTION("implicit SourceTerm" + execName)
    {
        fvcc::SourceTerm<TestType> sTerm(Operator::Type::Implicit, coeff, phi);
        auto ls = NeoN::la::createEmptyLinearSystem<
            TestType,
            NeoN::localIdx,
            NeoN::finiteVolume::cellCentred::SparsityPattern>(sp);
        sTerm.implicitOperation(ls);
        auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
        const auto& volView = vol.view();
        const auto& values = lsHost.matrix().values().view();

        for (auto ii = 0; ii < values.size(); ++ii)
        {
            REQUIRE(values[ii] - 2 * volView[0] * one<TestType>() == TestType(0.0));
        }
    }
}

}
