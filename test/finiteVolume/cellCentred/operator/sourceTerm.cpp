// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

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

    auto coeffBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> coeff(exec, "coeff", mesh, coeffBCs);
    fill(coeff.internalVector(), 2.0);
    fill(coeff.boundaryData().value(), 0.0);
    coeff.correctBoundaryConditions();

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    fill(phi.internalVector(), 10 * one<TestType>());
    fill(phi.boundaryData().value(), zero<TestType>());
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

        auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);

        sTerm.implicitOperation(ls);
        auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
        const auto& volView = vol.view();
        const auto& values = lsHost.matrix().values().view();

        for (auto ii = 0; ii < values.size(); ++ii)
        {
            REQUIRE(values[ii] - 2 * volView[0] * one<TestType>() == TestType(0.0));
        }
    }

    SECTION("implicit SuSp positive coefficient" + execName)
    {
        // coeff = +2 >= 0: SuSp behaves exactly like Sp — diagonal += coeff*vol,
        // rhs untouched (max(coeff,0) = coeff, min(coeff,0) = 0).
        fvcc::SourceTerm<TestType> sTerm(Operator::Type::Implicit, coeff, phi, /*suSp=*/true);

        auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);
        sTerm.implicitOperation(ls);
        auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
        const auto& volView = vol.view();
        const auto& values = lsHost.matrix().values().view();
        const auto& rhs = lsHost.rhs().view();

        REQUIRE(values[0] - 2 * volView[0] * one<TestType>() == TestType(0.0));
        REQUIRE(rhs[0] == zero<TestType>());
    }

    SECTION("implicit SuSp negative coefficient" + execName)
    {
        // coeff = -3 < 0: SuSp puts nothing on the diagonal (max(-3,0) = 0) and the
        // whole term on the rhs — rhs -= min(coeff,0)*vol*phi = +3*vol*phi.
        fill(coeff.internalVector(), -3.0);
        fvcc::SourceTerm<TestType> sTerm(Operator::Type::Implicit, coeff, phi, /*suSp=*/true);

        auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);
        sTerm.implicitOperation(ls);
        auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
        const auto& volView = vol.view();
        const auto& values = lsHost.matrix().values().view();
        const auto& rhs = lsHost.rhs().view();

        REQUIRE(values[0] == zero<TestType>());
        // phi = 10, so rhs = 3 * vol * 10 = 30 * vol
        REQUIRE(rhs[0] - 30 * volView[0] * one<TestType>() == TestType(0.0));
    }

    SECTION("SuSp splits the raw coefficient, not the scaled one" + execName)
    {
        // The expression scaling must act on the already-split term, i.e. the assembled
        // contribution has to be linear in it. If the split were taken on
        // scaling*coeff instead, a negative scaling would move a positive coefficient
        // from the diagonal onto the rhs rather than negating the diagonal entry.
        fvcc::SourceTerm<TestType> sTerm(Operator::Type::Implicit, coeff, phi, /*suSp=*/true);
        sTerm.getCoefficient() = dsl::Coeff(-1.0);

        auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);
        sTerm.implicitOperation(ls);
        auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
        const auto& volView = vol.view();
        const auto& values = lsHost.matrix().values().view();
        const auto& rhs = lsHost.rhs().view();

        // coeff = +2 is entirely implicit, so -1 * SuSp negates the diagonal entry and
        // leaves the rhs alone.
        REQUIRE(values[0] + 2 * volView[0] * one<TestType>() == TestType(0.0));
        REQUIRE(rhs[0] == zero<TestType>());
    }

    SECTION("SuSp cancels against itself under expression subtraction" + execName)
    {
        // susp(c, phi) - susp(c, phi) must assemble nothing at all. This is the property
        // the raw-coefficient split buys, and it holds for either sign of c.
        const auto c = GENERATE(NeoN::scalar {2.0}, NeoN::scalar {-3.0});
        fill(coeff.internalVector(), c);

        auto eqn = dsl::imp::susp<TestType>(coeff, phi) - dsl::imp::susp<TestType>(coeff, phi);

        auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);
        eqn.assembleSpatialOperator(ls);
        auto lsHost = ls.copyToHost();
        const auto& values = lsHost.matrix().values().view();
        const auto& rhs = lsHost.rhs().view();

        for (auto ii = 0; ii < values.size(); ++ii)
        {
            REQUIRE(values[ii] == zero<TestType>());
        }
        REQUIRE(rhs[0] == zero<TestType>());
    }
}

TEMPLATE_TEST_CASE("SourceTerm Su constructor", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = createSingleCellMesh(exec);

    auto coeffBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> coeff(exec, "coeff", mesh, coeffBCs);
    fill(coeff.internalVector(), 5 * one<TestType>());
    fill(coeff.boundaryData().value(), zero<TestType>());
    coeff.correctBoundaryConditions();

    SECTION("explicit Su" + execName)
    {
        fvcc::SourceTerm<TestType> sTerm(Operator::Type::Explicit, coeff);

        auto source = Vector<TestType>(exec, coeff.size(), zero<TestType>());
        sTerm.explicitOperation(source);

        auto exp = std::vector<TestType>(static_cast<size_t>(coeff.size()), 5 * one<TestType>());
        REQUIRE_THAT(source, Equals(exp, EqualInt()));
    }
}

}
