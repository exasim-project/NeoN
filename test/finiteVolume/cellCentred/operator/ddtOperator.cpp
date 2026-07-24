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

template<typename ValueType>
struct CreateVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<ValueType>> bcs {};
        for (auto patchi : std::vector<NeoN::localIdx> {0, 1, 2, 3})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", ValueType(2.0));
            bcs.push_back(fvcc::VolumeBoundary<ValueType>(mesh, dict, patchi));
        }
        NeoN::Field<ValueType> domainVector(
            mesh.exec(),
            NeoN::Vector<ValueType>(mesh.exec(), mesh.nCells(), one<ValueType>()),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<ValueType> vf(mesh.exec(), name, mesh, domainVector, bcs, db, "", "");

        return NeoN::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            fvcc::validateVectorDoc
        );
    }
};

TEMPLATE_TEST_CASE("DdtOperator", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Database db;
    auto mesh = createSingleCellMesh(exec);

    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "testVectorCollection");

    fvcc::VolumeField<TestType>& phi = fieldCollection.registerVector<fvcc::VolumeField<TestType>>(
        CreateVector<TestType> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );

    fill(phi.internalVector(), 10 * one<TestType>());
    fill(phi.boundaryData().value(), zero<TestType>());
    phi.correctBoundaryConditions();
    fill(oldTime(phi).internalVector(), -1.0 * one<TestType>());

    SECTION("explicit DdtOperator " + execName)
    {
        auto ddtOp = dsl::exp::ddt(phi);
        auto source = Vector<TestType>(exec, phi.size(), zero<TestType>());
        ddtOp.explicitOperation(source, 1.0, 0.5);

        const auto [vol, hostSource] = copyToHosts(mesh.cellVolumes(), source);
        const auto [volV, vals] = views(vol, hostSource);

        for (auto ii = 0; ii < vals.size(); ++ii)
        {
            // => (phi^{n + 1} - phi^{n})/dt*V => (10 -- 1)/.5*V = 22V
            REQUIRE(vals[ii] == volV[0] * TestType(22.0));
        }
    }

    SECTION("implicit DdtOperator (Euler) " + execName)
    {
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(phi)", std::string("BDF1"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);

        auto ddtOp = dsl::imp::ddt(phi);
        ddtOp.read(fvSchemes);
        ddtOp.implicitOperation(ls, 1.0, 0.5);

        const auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
        const auto [mtxValsV, volV, rhsV] = views(lsHost.matrix().values(), vol, lsHost.rhs());

        for (auto ii = 0; ii < mtxValsV.size(); ++ii)
        {
            // => 1/dt*V => 1/.5*V = 2V
            REQUIRE(mtxValsV[ii] == 2.0 * volV[0] * one<TestType>());
            // => phi^{n}/dt*V => -1/.5*V = -2V
            REQUIRE(rhsV[ii] == -2.0 * volV[0] * one<TestType>());
        }
    }

    SECTION("implicit DdtOperator backward (BDF2) " + execName)
    {
        // fvSchemes selecting backward
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(phi)", std::string("BDF2"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        auto ddtOp = dsl::imp::ddt(phi);
        ddtOp.read(fvSchemes);

        const scalar dt = 0.5;
        {
            auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);

            // ---------- Step 1: startup (Euler) ----------
            ddtOp.implicitOperation(ls, 1.0, dt);

            const auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
            const auto [mtxValsV, volV, rhsV] = views(lsHost.matrix().values(), vol, lsHost.rhs());

            for (auto ii = 0; ii < mtxValsV.size(); ++ii)
            {
                // => 1/dt*V => 1/.5*V = 2V
                REQUIRE(mtxValsV[ii] == (1.0 / dt) * volV[0] * one<TestType>());
                // => phi^{n}/dt*V => -1/.5*V = -2V
                REQUIRE(rhsV[ii] == (1.0 / dt) * (-1.0) * volV[0] * one<TestType>());
            }
        }
        {
            auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);

            // ---------- Step 2: true BDF2 ----------
            fill(oldTime(oldTime(phi)).internalVector(), -2.0 * one<TestType>());
            ddtOp.implicitOperation(ls, 1.5, dt);

            const auto [lsHost, vol] = copyToHosts(ls, mesh.cellVolumes());
            const auto [mtxValsV, volV, rhsV] = views(lsHost.matrix().values(), vol, lsHost.rhs());

            const scalar inv2dt = 1.0 / (2.0 * dt);
            for (auto ii = 0; ii < mtxValsV.size(); ++ii)
            {
                // BDF2 diagonal: 3/(2dt)
                REQUIRE(mtxValsV[ii] == (3.0 * inv2dt) * volV[0] * one<TestType>());
                // RHS: (4 phi^n - phi^{n-1})/(2dt)
                REQUIRE(rhsV[ii] == ((4.0 * (-1.0) - (-2.0)) * inv2dt) * volV[0] * one<TestType>());
            }
        }
    }
}

// DdtOperator::implicitOperation<SystemMatrixType> called directly (bypassing TemporalOperator's
// type erasure, same as SourceTerm's CSR-vs-ELL test) -- works for both scalar and Vec3, since
// the restriction to a fixed scalar-only ELL type is only at the TemporalOperator dispatch layer.
TEMPLATE_TEST_CASE("DdtOperator matches for CSR and ELL", "[template]", NeoN::scalar, NeoN::Vec3)
{
    using CSRMatrix = NeoN::la::CSRMatrix<TestType, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<TestType, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Ddt is diagonal-only (no faces involved), so a single-cell mesh already exercises the
    // comparison meaningfully -- matches every other CreateVector/Database-based DdtOperator
    // test in this file; that combination hasn't been exercised with a multi-cell mesh before.
    NeoN::Database db;
    auto mesh = createSingleCellMesh(exec);

    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "testVectorCollection");

    fvcc::VolumeField<TestType>& phi = fieldCollection.registerVector<fvcc::VolumeField<TestType>>(
        CreateVector<TestType> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );

    fill(phi.internalVector(), 10 * one<TestType>());
    fill(phi.boundaryData().value(), zero<TestType>());
    phi.correctBoundaryConditions();
    fill(oldTime(phi).internalVector(), -1.0 * one<TestType>());

    SECTION("diag() and rhs() match " + execName)
    {
        fvcc::DdtOperator<TestType> ddtOp(Operator::Type::Implicit, phi);

        auto csrLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, CSRMatrix>(mesh);
        auto ellLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, ELLMatrix>(mesh);

        ddtOp.implicitOperation(csrLs, 1.0, 0.5);
        ddtOp.implicitOperation(ellLs, 1.0, 0.5);

        REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag()));
        REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs()));
    }

    SECTION("diag() and rhs() match, BDF2 " + execName)
    {
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(phi)", std::string("BDF2"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        fvcc::DdtOperator<TestType> ddtOp(Operator::Type::Implicit, phi);
        ddtOp.read(fvSchemes);

        const scalar dt = 0.5;

        // Step 1: startup (Euler) -- level < 2, both formats fall back to bdf1Kernel.
        {
            auto csrLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, CSRMatrix>(mesh);
            auto ellLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, ELLMatrix>(mesh);
            ddtOp.implicitOperation(csrLs, 1.0, dt);
            ddtOp.implicitOperation(ellLs, 1.0, dt);
            REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag()));
            REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs()));
        }
        // Step 2: true BDF2, once a second old-time level exists.
        {
            fill(oldTime(oldTime(phi)).internalVector(), -2.0 * one<TestType>());

            auto csrLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, CSRMatrix>(mesh);
            auto ellLs = NeoN::la::createEmptyLinearSystem<TestType, TestType, ELLMatrix>(mesh);
            ddtOp.implicitOperation(csrLs, 1.5, dt);
            ddtOp.implicitOperation(ellLs, 1.5, dt);
            REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag()));
            REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs()));
        }
    }
}

// Full vertical slice for the temporal side: dsl::imp::ddt() (the production entry point) goes
// through TemporalOperator's new ELL dispatch (implicitOperationELL, mirroring SpatialOperator's)
// via Expression::assemble<scalar, ELLMatrix<scalar, localIdx>>() -- proves TemporalOperator's
// generalization actually works end to end, not just that DdtOperator's own kernel does.
TEST_CASE("Expression assembles ddt into ELL via TemporalOperator")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Database db;
    auto mesh = createSingleCellMesh(exec);

    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "testVectorCollection");

    fvcc::VolumeField<scalar>& phi = fieldCollection.registerVector<fvcc::VolumeField<scalar>>(
        CreateVector<scalar> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );

    fill(phi.internalVector(), 10 * one<scalar>());
    fill(phi.boundaryData().value(), zero<scalar>());
    phi.correctBoundaryConditions();
    fill(oldTime(phi).internalVector(), -1.0 * one<scalar>());

    NeoN::Dictionary fvSchemes;
    NeoN::Dictionary ddtSchemes;
    ddtSchemes.insert("ddt(phi)", std::string("BDF1"));
    fvSchemes.insert("ddtSchemes", ddtSchemes);

    dsl::Expression<scalar> expr(dsl::imp::ddt(phi));
    expr.read(fvSchemes);

    auto csrLs = expr.assemble<scalar, CSRMatrix>(mesh, 1.0, 0.5);
    auto ellLs = expr.assemble<scalar, ELLMatrix>(mesh, 1.0, 0.5);

    REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag()));
    REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs()));
}

// Segregated vector-solve form (scalar matrix, Vec3 rhs) of "DdtOperator matches for CSR and
// ELL" above -- DdtOperator::implicitOperation<SystemMatrixType> called directly, bypassing
// TemporalOperator's type erasure.
TEST_CASE("DdtOperator matches for CSR and ELL, segregated")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Database db;
    auto mesh = createSingleCellMesh(exec);

    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "testVectorCollection");

    fvcc::VolumeField<Vec3>& phi = fieldCollection.registerVector<fvcc::VolumeField<Vec3>>(
        CreateVector<Vec3> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );

    fill(phi.internalVector(), 10 * one<Vec3>());
    fill(phi.boundaryData().value(), zero<Vec3>());
    phi.correctBoundaryConditions();
    fill(oldTime(phi).internalVector(), -1.0 * one<Vec3>());

    SECTION("diag() and rhs() match " + execName)
    {
        fvcc::DdtOperator<Vec3> ddtOp(Operator::Type::Implicit, phi);

        auto csrLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, CSRMatrix>(mesh);
        auto ellLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, ELLMatrix>(mesh);

        ddtOp.implicitOperation(csrLs, 1.0, 0.5);
        ddtOp.implicitOperation(ellLs, 1.0, 0.5);

        REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag()));
        REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs()));
    }

    SECTION("diag() and rhs() match, BDF2 " + execName)
    {
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(phi)", std::string("BDF2"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        fvcc::DdtOperator<Vec3> ddtOp(Operator::Type::Implicit, phi);
        ddtOp.read(fvSchemes);

        const scalar dt = 0.5;

        // Step 1: startup (Euler) -- level < 2, both formats fall back to bdf1KernelScalarMtx.
        {
            auto csrLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, CSRMatrix>(mesh);
            auto ellLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, ELLMatrix>(mesh);
            ddtOp.implicitOperation(csrLs, 1.0, dt);
            ddtOp.implicitOperation(ellLs, 1.0, dt);
            REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag()));
            REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs()));
        }
        // Step 2: true BDF2, once a second old-time level exists.
        {
            fill(oldTime(oldTime(phi)).internalVector(), -2.0 * one<Vec3>());

            auto csrLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, CSRMatrix>(mesh);
            auto ellLs = NeoN::la::createEmptyLinearSystem<scalar, Vec3, ELLMatrix>(mesh);
            ddtOp.implicitOperation(csrLs, 1.5, dt);
            ddtOp.implicitOperation(ellLs, 1.5, dt);
            REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag()));
            REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs()));
        }
    }
}

// Full vertical slice for the segregated form: dsl::imp::ddt() with a Vec3 field goes through
// TemporalOperator's segregated-ELL dispatch (HasTemporalImplicitOperatorScalarMtxELL /
// implicitOperationScalarMtxELL) via Expression<Vec3>::assemble<scalar, ELLMatrix<scalar,
// localIdx>>() -- the gap that used to break compilation for every Vec3 expression assembling
// into ELL, not just ddt-containing ones (see temporalOperator.hpp).
TEST_CASE("Expression assembles ddt into ELL via TemporalOperator, segregated")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Database db;
    auto mesh = createSingleCellMesh(exec);

    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "testVectorCollection");

    fvcc::VolumeField<Vec3>& phi = fieldCollection.registerVector<fvcc::VolumeField<Vec3>>(
        CreateVector<Vec3> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );

    fill(phi.internalVector(), 10 * one<Vec3>());
    fill(phi.boundaryData().value(), zero<Vec3>());
    phi.correctBoundaryConditions();
    fill(oldTime(phi).internalVector(), -1.0 * one<Vec3>());

    NeoN::Dictionary fvSchemes;
    NeoN::Dictionary ddtSchemes;
    ddtSchemes.insert("ddt(phi)", std::string("BDF1"));
    fvSchemes.insert("ddtSchemes", ddtSchemes);

    dsl::Expression<Vec3> expr(dsl::imp::ddt(phi));
    expr.read(fvSchemes);

    auto csrLs = expr.assemble<scalar, CSRMatrix>(mesh, 1.0, 0.5);
    auto ellLs = expr.assemble<scalar, ELLMatrix>(mesh, 1.0, 0.5);

    REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag()));
    REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs()));
}

}
