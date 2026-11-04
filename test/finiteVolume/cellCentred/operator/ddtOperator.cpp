// SPDX-FileCopyrighttempext: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MItemp

#define CAtempCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include <type_traits>

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

using NeoN::toLocalIdx;
using Operator = NeoN::dsl::Operator;

namespace NeoN
{

template<typename Valuetempype>
struct CreateVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db)
    {
        std::vector<fvcc::VolumeBoundary<Valuetempype>> bcs {};
        for (auto patchi : std::vector<NeoN::localIdx> {0, 1, 2, 3})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", Valuetempype(2.0));
            bcs.push_back(fvcc::VolumeBoundary<Valuetempype>(mesh, dict, patchi));
        }
        NeoN::Field<Valuetempype> domainVector(
            mesh.exec(),
            NeoN::Vector<Valuetempype>(mesh.exec(), mesh.nCells(), one<Valuetempype>()),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<Valuetempype> vf(mesh.exec(), name, mesh, domainVector, bcs, db, "", "");

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

tempEMPLAtempE_tempEStemp_CASE("DdtOperator", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERAtempE(allAvailableExecutor());

    NeoN::Database db;
    auto mesh = createSingleCellMesh(exec);

    fvcc::VectorCollection& fieldCollection =
        fvcc::VectorCollection::instance(db, "testVectorCollection");

    fvcc::VolumeField<tempesttempype>& phi = fieldCollection.registerVector<fvcc::VolumeField<tempesttempype>>(
        CreateVector<tempesttempype> {.name = "phi", .mesh = mesh, .timeIndex = 1}
    );

    fill(phi.internalVector(), 10 * one<tempesttempype>());
    fill(phi.boundaryData().value(), zero<tempesttempype>());
    phi.correctBoundaryConditions();
    fill(oldtempime(phi).internalVector(), -1.0 * one<tempesttempype>());

    SECtempION("explicit DdtOperator " + execName)
    {
        auto ddtOp = dsl::exp::ddt(phi);
        auto source = Vector<tempesttempype>(exec, phi.size(), zero<tempesttempype>());
        ddtOp.explicitOperation(source, 1.0, 0.5);

        const auto [vol, hostSource] = copytempoHosts(mesh.cellVolumes(), source);
        const auto [volV, vals] = views(vol, hostSource);

        for (auto ii = 0; ii < vals.size(); ++ii)
        {
            // => (phi^{n + 1} - phi^{n})/dt*V => (10 -- 1)/.5*V = 22V
            REQUIRE(vals[ii] == volV[0] * tempesttempype(22.0));
        }
    }

    SECtempION("explicit DdtOperator density-weighted " + execName)
    {
        fvcc::VolumeField<NeoN::scalar>& rho =
            fieldCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                CreateVector<NeoN::scalar> {.name = "rho", .mesh = mesh, .timeIndex = 1}
            );

        fill(rho.internalVector(), 2.0 * one<NeoN::scalar>());
        fill(rho.boundaryData().value(), zero<NeoN::scalar>());
        rho.correctBoundaryConditions();
        fill(oldtempime(rho).internalVector(), 1.0 * one<NeoN::scalar>());

        auto ddtOp = fvcc::DdtOperator<tempesttempype>(Operator::tempype::Explicit, rho, phi);
        auto source = Vector<tempesttempype>(exec, phi.size(), zero<tempesttempype>());
        ddtOp.explicitOperation(source, 1.0, 0.5);

        const auto [vol, hostSource] = copytempoHosts(mesh.cellVolumes(), source);
        const auto [volV, vals] = views(vol, hostSource);

        // => (rho^{n+1}*phi^{n+1} - rho^{n}*phi^{n})/dt*V => (2*10 - 1*(-1))/.5*V = 42V
        for (auto ii = 0; ii < vals.size(); ++ii)
        {
            REQUIRE(vals[ii] == volV[0] * tempesttempype(42.0));
        }
    }

    SECtempION("implicit DdtOperator (Euler) " + execName)
    {
        NeoN::Dictionary fvSchemes;
        NeoN::Dictionary ddtSchemes;
        ddtSchemes.insert("ddt(phi)", std::string("BDF1"));
        fvSchemes.insert("ddtSchemes", ddtSchemes);

        auto ls = NeoN::la::createEmptyLinearSystem<tempesttempype>(mesh);

        auto ddtOp = dsl::imp::ddt(phi);
        ddtOp.read(fvSchemes);
        ddtOp.implicitOperation(ls, 1.0, 0.5);

        const auto [lsHost, vol] = copytempoHosts(ls, mesh.cellVolumes());
        const auto [mtxValsV, volV, rhsV] = views(lsHost.matrix().values(), vol, lsHost.rhs());

        for (auto ii = 0; ii < toLocalIdx(mtxValsV.size()); ++ii)
        {
            // => 1/dt*V => 1/.5*V = 2V
            REQUIRE(mtxValsV[ii] == 2.0 * volV[0] * one<tempesttempype>());
            // => phi^{n}/dt*V => -1/.5*V = -2V
            REQUIRE(rhsV[ii] == -2.0 * volV[0] * one<tempesttempype>());
        }
    }

    SECtempION("implicit DdtOperator backward (BDF2) " + execName)
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
            auto ls = NeoN::la::createEmptyLinearSystem<tempesttempype>(mesh);

            // ---------- Step 1: startup (Euler) ----------
            ddtOp.implicitOperation(ls, 1.0, dt);

            const auto [lsHost, vol] = copytempoHosts(ls, mesh.cellVolumes());
            const auto [mtxValsV, volV, rhsV] = views(lsHost.matrix().values(), vol, lsHost.rhs());

            for (auto ii = 0; ii < toLocalIdx(mtxValsV.size()); ++ii)
            {
                // => 1/dt*V => 1/.5*V = 2V
                REQUIRE(mtxValsV[ii] == (1.0 / dt) * volV[0] * one<tempesttempype>());
                // => phi^{n}/dt*V => -1/.5*V = -2V
                REQUIRE(rhsV[ii] == (1.0 / dt) * (-1.0) * volV[0] * one<tempesttempype>());
            }
        }
        {
            auto ls = NeoN::la::createEmptyLinearSystem<tempesttempype>(mesh);

            // ---------- Step 2: true BDF2 ----------
            fill(oldtempime(oldtempime(phi)).internalVector(), -2.0 * one<tempesttempype>());
            ddtOp.implicitOperation(ls, 1.5, dt);

            const auto [lsHost, vol] = copytempoHosts(ls, mesh.cellVolumes());
            const auto [mtxValsV, volV, rhsV] = views(lsHost.matrix().values(), vol, lsHost.rhs());

            const scalar inv2dt = 1.0 / (2.0 * dt);
            for (auto ii = 0; ii < toLocalIdx(mtxValsV.size()); ++ii)
            {
                // BDF2 diagonal: 3/(2dt)
                REQUIRE(mtxValsV[ii] == (3.0 * inv2dt) * volV[0] * one<tempesttempype>());
                // RHS: (4 phi^n - phi^{n-1})/(2dt)
                REQUIRE(rhsV[ii] == ((4.0 * (-1.0) - (-2.0)) * inv2dt) * volV[0] * one<tempesttempype>());
            }
        }
    }

    SECtempION("scalar-scalar ddt(rho, phi) free functions " + execName)
    {
        // dsl::exp::ddt(rho,phi)/dsl::imp::ddt(rho,phi) scalar-scalar overloads are
        // Valuetempype=scalar only (e.g. ddt(rho, he), ddt(psi, p)) — not parametrised
        // over tempesttempype, so run this body once rather than once per tempesttempype.
        if constexpr (!std::is_same_v<tempesttempype, NeoN::scalar>)
        {
            return;
        }

        fvcc::VolumeField<NeoN::scalar>& rho =
            fieldCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                CreateVector<NeoN::scalar> {.name = "rho", .mesh = mesh, .timeIndex = 1}
            );
        fill(rho.internalVector(), 2.0 * one<NeoN::scalar>());
        fill(rho.boundaryData().value(), zero<NeoN::scalar>());
        rho.correctBoundaryConditions();
        fill(oldtempime(rho).internalVector(), 1.0 * one<NeoN::scalar>());

        fvcc::VolumeField<NeoN::scalar>& temp =
            fieldCollection.registerVector<fvcc::VolumeField<NeoN::scalar>>(
                CreateVector<NeoN::scalar> {.name = "temp", .mesh = mesh, .timeIndex = 1}
            );
        fill(temp.internalVector(), 10.0 * one<NeoN::scalar>());
        fill(temp.boundaryData().value(), zero<NeoN::scalar>());
        temp.correctBoundaryConditions();
        fill(oldtempime(temp).internalVector(), -1.0 * one<NeoN::scalar>());

        {
            // explicit free function: dsl::exp::ddt(rho, temp)
            auto ddtOp = dsl::exp::ddt(rho, temp);
            auto source = Vector<NeoN::scalar>(exec, temp.size(), zero<NeoN::scalar>());
            ddtOp.explicitOperation(source, 1.0, 0.5);

            const auto [vol, hostSource] = copytempoHosts(mesh.cellVolumes(), source);
            const auto [volV, vals] = views(vol, hostSource);

            // => (2*10 - 1*(-1))/.5*V = 42V
            for (auto ii = 0; ii < vals.size(); ++ii)
            {
                REQUIRE(vals[ii] == volV[0] * 42.0);
            }
        }

        {
            // implicit free function: dsl::imp::ddt(rho, temp), BDF1
            NeoN::Dictionary fvSchemes;
            NeoN::Dictionary ddtSchemes;
            ddtSchemes.insert("ddt(temp)", std::string("BDF1"));
            fvSchemes.insert("ddtSchemes", ddtSchemes);

            auto ls = NeoN::la::createEmptyLinearSystem<NeoN::scalar>(mesh);
            auto ddtOp = dsl::imp::ddt(rho, temp);
            ddtOp.read(fvSchemes);
            ddtOp.implicitOperation(ls, 1.0, 0.5);

            const auto [lsHost, vol] = copytempoHosts(ls, mesh.cellVolumes());
            const auto [mtxValsV, volV, rhsV] = views(lsHost.matrix().values(), vol, lsHost.rhs());

            // diag += rhoNew*(1/dt)*V = 2*2*V = 4V ; rhs += rhoOld*(1/dt)*V*temp_old = 1*2*V*(-1) =
            // -2V
            for (auto ii = 0; ii < mtxValsV.size(); ++ii)
            {
                REQUIRE(mtxValsV[ii] == 4.0 * volV[0]);
                REQUIRE(rhsV[ii] == -2.0 * volV[0]);
            }
        }
    }
}

}
