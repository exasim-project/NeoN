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
}

// Same Sp assembly, run through the templated implicitOperation() directly against both a CSR
// and an ELL system (bypassing the CSR-only dsl::SpatialOperator interface) -- proves the
// operator body itself is format-generic, independent of the DSL generalization that's still
// pending.
TEMPLATE_TEST_CASE("SourceTerm matches for CSR and ELL", "[template]", NeoN::scalar, NeoN::Vec3)
{
    using CSRMatrix = NeoN::la::CSRMatrix<TestType, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<TestType, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);

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

    SECTION("diag() matches " + execName)
    {
        fvcc::SourceTerm<TestType> sTerm(Operator::Type::Implicit, coeff, phi);

        auto csrLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, CSRMatrix>(mesh);
        auto ellLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, ELLMatrix>(mesh);

        sTerm.template implicitOperation<CSRMatrix>(csrLs);
        sTerm.template implicitOperation<ELLMatrix>(ellLs);

        REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag()));

        // ELL's sparsity here is the full mesh-connectivity stencil (diag + neighbours), not
        // diag-only, so it has both real off-diagonal slots and padding. Sp only ever writes
        // ma.diagIdx(): check every other slot -- padding included -- stayed exactly zero.
        auto hostLs = ellLs.copyToExecutor(SerialExecutor());
        auto ma = hostLs.matrix().faceToMatrixView();
        auto valuesHostV = hostLs.matrix().values().view();

        std::vector<localIdx> diagOffsets(static_cast<std::size_t>(nCells));
        for (localIdx celli = 0; celli < nCells; ++celli)
        {
            diagOffsets[static_cast<std::size_t>(celli)] = ma.diagIdx(celli);
        }

        for (localIdx i = 0; i < valuesHostV.size(); ++i)
        {
            bool isDiagSlot =
                std::find(diagOffsets.begin(), diagOffsets.end(), i) != diagOffsets.end();
            if (!isDiagSlot)
            {
                REQUIRE(valuesHostV[i] == zero<TestType>());
            }
        }
    }
}

#if NF_WITH_GINKGO
// Full vertical slice: a normal NeoN DSL expression (dsl::imp::source, the production entry
// point) assembles into an ELL system via Expression::assemble<AssemblyType, SystemMatrixType>()
// and solves through the same la::Solver interface production code holds -- not by calling
// SourceTerm::implicitOperation() or GinkgoSolver::solveImpl() directly, unlike every other ELL
// test so far. dsl::solve() itself now supports ELL too (see test/dsl/solver.cpp); this test
// predates that and calls Expression::assemble() and Solver::solve() directly, which is still a
// meaningful vertical slice on its own.
TEST_CASE("Expression assembles ELL and solves through Solver, matches CSR")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto mesh = create1DUniformMesh(exec, nCells);

    auto coeffBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> coeff(exec, "coeff", mesh, coeffBCs);
    fill(coeff.internalVector(), 2.0);
    fill(coeff.boundaryData().value(), 0.0);
    coeff.correctBoundaryConditions();

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "sf", mesh, volumeBCs);
    fill(phi.internalVector(), zero<scalar>());
    fill(phi.boundaryData().value(), zero<scalar>());
    phi.correctBoundaryConditions();

    dsl::Expression<scalar> expr(dsl::imp::source(coeff, phi));

    // The mesh-owning overload -- Expression itself selects SystemMatrixType and constructs
    // the system via createEmptyLinearSystem, rather than assembling into a caller-provided
    // one. Our source term is Implicit-type only, so assembleExplicitSource's subtraction is a
    // net no-op here; rhs is filled directly afterward for a genuinely solvable system.
    auto csrLs = expr.assemble<scalar, CSRMatrix>(mesh, 0.0, 0.0);
    auto ellLs = expr.assemble<scalar, ELLMatrix>(mesh, 0.0, 0.0);
    fill(csrLs.rhs(), 1.0);
    fill(ellLs.rhs(), 1.0);

    Dictionary solverDict {
        {{"solver", std::string {"Ginkgo"}},
         {"type", "solver::Cg"},
         {"criteria", Dictionary {{{"iteration", 200}, {"relative_residual_norm", 1e-12}}}}}
    };
    auto solver = NeoN::la::Solver(exec, solverDict);

    Vector<scalar> xCsr(exec, nCells, 0.0);
    Vector<scalar> xEll(exec, nCells, 0.0);
    auto csrStats = solver.solve(csrLs, xCsr);
    auto ellStats = solver.solve(ellLs, xEll);

    REQUIRE_FALSE(csrStats.entries.empty());
    REQUIRE_FALSE(ellStats.entries.empty());
    REQUIRE(csrStats.entries.front().numIter > 0);
    REQUIRE(ellStats.entries.front().numIter > 0);

    REQUIRE_THAT(xEll, Equals(xCsr, Approx {1e-8}));
}
#endif

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
