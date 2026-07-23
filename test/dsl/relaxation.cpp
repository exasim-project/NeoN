// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Vector;
using NeoN::Vec3;
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace dsl = NeoN::dsl;

// ---------------------------------------------------------------------------
// Host-side componentwise helpers (mirror the device helpers in dsl/solver.hpp)
// so the expected values can be hand-computed on the host for every component.
// ---------------------------------------------------------------------------
namespace
{

scalar hostCompMag(scalar v) { return std::abs(v); }
Vec3 hostCompMag(const Vec3& v) { return Vec3(std::abs(v[0]), std::abs(v[1]), std::abs(v[2])); }

scalar hostCompMax(scalar a, scalar b) { return std::max(a, b); }
Vec3 hostCompMax(const Vec3& a, const Vec3& b)
{
    return Vec3(std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2]));
}

scalar hostCopySign(scalar mag, scalar s) { return (s >= 0) ? mag : -mag; }
Vec3 hostCopySign(const Vec3& mag, const Vec3& s)
{
    return Vec3(hostCopySign(mag[0], s[0]), hostCopySign(mag[1], s[1]), hostCopySign(mag[2], s[2]));
}

// componentwise sign sentinel: +1 / -1 per component, used for the [sign] section.
int sgn(scalar v) { return (v > 0) - (v < 0); }

// Find the flat CSR index of cell c's diagonal entry (col == row) on a host-copied
// matrix view by scanning the row. This is the executor-independent definition and
// avoids depending on FaceToMatrixAddress::copyToHost (whose diagOffset round-trips
// inconsistently for CPU/GPU-sourced systems -- a NeoN infra quirk, out of scope here).
template<typename HostMatrixView>
localIdx hostDiagIdx(const HostMatrixView& m, localIdx c)
{
    for (localIdx idx = m.sparsity.rowOffs[c]; idx < m.sparsity.rowOffs[c + 1]; ++idx)
    {
        if (m.sparsity.colIdxs[idx] == c)
        {
            return idx;
        }
    }
    return -1;
}

// A non-trivial constant value with all three Vec3 components distinct, so the
// Vec3 path is exercised per-component.
template<typename T>
T sampleValue(scalar base);
template<>
scalar sampleValue<scalar>(scalar base)
{
    return base;
}
template<>
Vec3 sampleValue<Vec3>(scalar base)
{
    return Vec3(base, 2.0 * base, -3.0 * base);
}

} // namespace

// ---------------------------------------------------------------------------
// Helper: build a synthetic LinearSystem + matching VolumeField on a real
// (tiny) mesh so the kernel has BOTH a FaceToMatrixAddress (createEmptyLinearSystem
// builds one) AND a faceOwners() topology. A raw CSRMatrix(values,...) ctor leaves
// faceToMatrixAddress() null, so a real mesh is mandatory for the kernel under test.
//
// The boundaryMatrix from createEmptyLinearSystem is zero-initialised, so the
// per-cell boundaryDiag is zero for every cell — the sign/noop/clamp sections are
// therefore independent of the boundary path, which keeps the hand-computed
// expected values simple. The diagonal is built NEGATIVE (NeoN convention) with an
// asymmetric, known off-diagonal pattern so componentCopySign and the clamp are
// genuinely exercised (defeats the 1D-uniform blind spot).
// ---------------------------------------------------------------------------
TEMPLATE_TEST_CASE(
    "applyMatrixRelaxation",
    "[relaxation][sign][noop][clamp][fixedpoint][boundary]",
    NeoN::scalar,
    NeoN::Vec3
)
{
    using ValueType = TestType;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 3;
    auto mesh = NeoN::create1DUniformMesh(exec, nCells);

    // Boundary conditions for the companion VolumeField (one per patch; values
    // are irrelevant — only the mesh topology / internalVector are read).
    std::vector<fvcc::VolumeBoundary<ValueType>> bcs {};
    for (localIdx patchi = 0; patchi < mesh.nBoundaries(); ++patchi)
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", NeoN::zero<ValueType>());
        bcs.push_back(fvcc::VolumeBoundary<ValueType>(mesh, dict, patchi));
    }

    // Build the synthetic CSR by writing through the view: a NEGATIVE diagonal and
    // an asymmetric off-diagonal pattern. Off-diagonal magnitude per cell is chosen
    // so cell 0 is strongly diagonally dominant (clamp keeps |D_internal|) and cell 2
    // is off-diagonal dominant (clamp uses sumMagOffDiag -- the
    // "sumOff wins" sub-case). Returns the populated LinearSystem + solution field.
    auto buildSystem =
        [&](std::vector<scalar> diagBase, std::vector<scalar> offBase, std::vector<scalar> psiBase)
    {
        auto ls = NeoN::la::createEmptyLinearSystem<ValueType>(mesh);

        auto lsView = ls.view();
        auto matrix = lsView.matrix;
        const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
        const auto [rowOffs, colIdxs] = NeoN::views(ls.matrix().rowOffs(), ls.matrix().colIdxs());

        Vector<ValueType> diagV(exec, nCells);
        Vector<ValueType> offV(exec, nCells);
        {
            auto dh = diagV.copyToHost();
            auto oh = offV.copyToHost();
            for (localIdx c = 0; c < nCells; ++c)
            {
                dh.view()[c] = sampleValue<ValueType>(diagBase[static_cast<std::size_t>(c)]);
                oh.view()[c] = sampleValue<ValueType>(offBase[static_cast<std::size_t>(c)]);
            }
            diagV = dh.copyToExecutor(exec);
            offV = oh.copyToExecutor(exec);
        }
        const auto diagVals = diagV.view();
        const auto offVals = offV.view();

        NeoN::parallelFor(
            exec,
            {0, nCells},
            NEON_LAMBDA(const localIdx celli) {
                const auto diagIdx = ma.diagIdx(celli);
                for (localIdx idx = rowOffs[celli]; idx < rowOffs[celli + 1]; ++idx)
                {
                    if (idx == diagIdx)
                    {
                        matrix.values[idx] = diagVals[celli];
                    }
                    else
                    {
                        matrix.values[idx] = offVals[celli];
                    }
                }
            },
            "buildSyntheticCSR"
        );
        // Fence so the (possibly async) build kernel finishes reading diagVals/offVals
        // before those local Vectors are destroyed when this lambda returns. Without
        // this, the CPU-threads / GPU backends read freed memory (Serial is synchronous,
        // so it accidentally passes).
        NeoN::fence(exec);

        Vector<ValueType> psi(exec, nCells);
        {
            auto ph = psi.copyToHost();
            for (localIdx c = 0; c < nCells; ++c)
            {
                ph.view()[c] = sampleValue<ValueType>(psiBase[static_cast<std::size_t>(c)]);
            }
            psi = ph.copyToExecutor(exec);
        }
        fvcc::VolumeField<ValueType> solution(exec, "psi", mesh, psi, bcs);

        return std::make_pair(std::move(ls), std::move(solution));
    };

    SECTION("sign preservation on " + execName)
    {
        // Negative diagonal, asymmetric off-diagonals. alpha = 0.5.
        auto [ls, solution] = buildSystem({-4.0, -5.0, -1.0}, {1.0, -2.0, 3.0}, {0.7, -0.3, 0.5});

        auto before = ls.copyToHost();
        auto beforeView = before.view();
        std::vector<ValueType> diagBefore(nCells);
        for (localIdx c = 0; c < nCells; ++c)
        {
            diagBefore[static_cast<std::size_t>(c)] =
                beforeView.matrix.values[hostDiagIdx(beforeView.matrix, c)];
        }

        dsl::applyMatrixRelaxation(ls, solution, 0.5);

        auto after = ls.copyToHost();
        auto afterView = after.view();
        for (localIdx c = 0; c < nCells; ++c)
        {
            const ValueType d0 = diagBefore[static_cast<std::size_t>(c)];
            const ValueType d1 = afterView.matrix.values[hostDiagIdx(afterView.matrix, c)];
            INFO("cell " << c << " diag before/after");
            if constexpr (std::is_same_v<ValueType, scalar>)
            {
                REQUIRE(sgn(d1) == sgn(d0));
            }
            else
            {
                REQUIRE(sgn(d1[0]) == sgn(d0[0]));
                REQUIRE(sgn(d1[1]) == sgn(d0[1]));
                REQUIRE(sgn(d1[2]) == sgn(d0[2]));
            }
        }
    }

    SECTION("alpha==1 is a bitwise no-op on " + execName)
    {
        auto [ls, solution] = buildSystem({-4.0, -5.0, -1.0}, {1.0, -2.0, 3.0}, {0.7, -0.3, 0.5});

        // Give rhs a non-trivial value so a regressed guard touching rhs is caught.
        {
            auto rhsH = ls.rhs().copyToHost();
            for (localIdx c = 0; c < nCells; ++c)
            {
                rhsH.view()[c] = sampleValue<ValueType>(static_cast<scalar>(c) + 1.5);
            }
            ls.rhs() = rhsH.copyToExecutor(exec);
        }

        auto before = ls.copyToHost();
        auto beforeView = before.view();
        std::vector<ValueType> valuesBefore(static_cast<std::size_t>(beforeView.matrix.values.size()
        ));
        for (localIdx i = 0; i < beforeView.matrix.values.size(); ++i)
        {
            valuesBefore[static_cast<std::size_t>(i)] = beforeView.matrix.values[i];
        }
        std::vector<ValueType> rhsBefore(static_cast<std::size_t>(nCells));
        for (localIdx c = 0; c < nCells; ++c)
        {
            rhsBefore[static_cast<std::size_t>(c)] = beforeView.rhs[c];
        }

        dsl::applyMatrixRelaxation(ls, solution, 1.0);

        auto after = ls.copyToHost();
        auto afterView = after.view();
        // BITWISE equality (==, not Approx): a diff means the early-return regressed.
        for (localIdx i = 0; i < afterView.matrix.values.size(); ++i)
        {
            REQUIRE(afterView.matrix.values[i] == valuesBefore[static_cast<std::size_t>(i)]);
        }
        for (localIdx c = 0; c < nCells; ++c)
        {
            REQUIRE(afterView.rhs[c] == rhsBefore[static_cast<std::size_t>(c)]);
        }

        // alpha == 0 must also be a bitwise no-op (Discretion alpha<=0 guard).
        dsl::applyMatrixRelaxation(ls, solution, 0.0);
        auto after0 = ls.copyToHost();
        auto after0View = after0.view();
        for (localIdx i = 0; i < after0View.matrix.values.size(); ++i)
        {
            REQUIRE(after0View.matrix.values[i] == valuesBefore[static_cast<std::size_t>(i)]);
        }
    }

    SECTION("nonzero boundaryDiag does not perturb the augmented-diag scaling on " + execName)
    {
        // The other sections build on createEmptyLinearSystem whose boundaryMatrix is
        // zero-initialised (boundaryDiag == 0), so the boundary path is untested there. Here we
        // populate the boundaryMatrix COO with NONZERO values through the view so each owner cell
        // has boundaryDiag != 0.
        //
        // The kernel relaxes the WHOLE augmented diagonal (boundary already baked into D_aug at
        // assembly) by 1/alpha and does NOT reconstruct/re-add the boundary diagonal, so the
        // relaxed augmented diagonal must satisfy
        //   matrix.diag(cell) == copySign(max(|D_aug|, sumOff)/alpha, D_aug)
        // INDEPENDENT of boundaryDiag.
        const scalar alpha = 0.7;
        const scalar invAlpha = 1.0 / alpha;

        std::vector<scalar> diagBase {-4.0, -5.0, -6.0};
        std::vector<scalar> offBase {1.0, 1.0, 1.0};
        std::vector<scalar> psiBase {0.0, 0.0, 0.0}; // psi=0: isolate the diagonal algebra
        auto [ls, solution] = buildSystem(diagBase, offBase, psiBase);

        // Populate the boundaryMatrix COO with nonzero values (one per boundary face). The
        // owner cell of each face is stored in the COO sparsity rowOffs (same convention
        // removeBoundaryContributions reads). create1DUniformMesh has real boundary faces
        // (the two end cells), so nBoundaryFaces > 0 here.
        const localIdx nbf = ls.boundaryMatrix().values().size();
        REQUIRE(nbf > 0); // create1DUniformMesh must expose physical boundary faces
        {
            auto bvH = ls.boundaryMatrix().values().copyToHost();
            for (localIdx f = 0; f < nbf; ++f)
            {
                // Distinct, sign-mixed boundary contributions so the per-component Vec3
                // path and the +/- reconstruct are genuinely exercised.
                bvH.view()[f] = sampleValue<ValueType>(0.25 * (static_cast<scalar>(f) + 1.0));
            }
            ls.boundaryMatrix().values() = bvH.copyToExecutor(exec);
        }

        // Capture D_aug (boundary-baked) and sumOff from the assembled CSR before relaxation.
        auto before = ls.copyToHost();
        auto beforeView = before.view();
        std::vector<ValueType> dAug(nCells), sumOff(nCells, NeoN::zero<ValueType>());
        for (localIdx c = 0; c < nCells; ++c)
        {
            const auto diagIdx = hostDiagIdx(beforeView.matrix, c);
            dAug[static_cast<std::size_t>(c)] = beforeView.matrix.values[diagIdx];
            for (localIdx idx = beforeView.matrix.sparsity.rowOffs[c];
                 idx < beforeView.matrix.sparsity.rowOffs[c + 1];
                 ++idx)
            {
                if (idx != diagIdx)
                {
                    sumOff[static_cast<std::size_t>(c)] =
                        sumOff[static_cast<std::size_t>(c)]
                        + hostCompMag(beforeView.matrix.values[idx]);
                }
            }
        }

        dsl::applyMatrixRelaxation(ls, solution, alpha);

        // The kernel scales the augmented diagonal (boundary contributions baked into D_aug at
        // assembly) by 1/alpha under the dominance clamp. The relaxed augmented diagonal equals
        //   copySign(max(|D_aug|, sumOff)/alpha, D_aug)
        // independent of the nonzero boundaryMatrix populated above.
        auto relaxedH = ls.copyToHost();
        auto relaxedView = relaxedH.view();
        for (localIdx c = 0; c < nCells; ++c)
        {
            const ValueType dDom = hostCompMax(
                hostCompMag(dAug[static_cast<std::size_t>(c)]), sumOff[static_cast<std::size_t>(c)]
            );
            const ValueType expected =
                hostCopySign(invAlpha * dDom, dAug[static_cast<std::size_t>(c)]);
            const ValueType got = relaxedView.matrix.values[hostDiagIdx(relaxedView.matrix, c)];
            INFO("nonzero-boundary cell " << c);
            if constexpr (std::is_same_v<ValueType, scalar>)
            {
                REQUIRE(got == Catch::Approx(expected).margin(1e-12));
            }
            else
            {
                REQUIRE(got[0] == Catch::Approx(expected[0]).margin(1e-12));
                REQUIRE(got[1] == Catch::Approx(expected[1]).margin(1e-12));
                REQUIRE(got[2] == Catch::Approx(expected[2]).margin(1e-12));
            }
        }
    }

    SECTION("clamp formula (dominant + sumOff-wins) on " + execName)
    {
        const scalar alpha = 0.5;
        const scalar invAlpha = 1.0 / alpha;

        // cell 0: |D_internal| = 4 > sumOff (2*|1| = 2)        -> diagonal dominant branch
        // cell 1: |D_internal| = 5 > sumOff (3*|2| = 6)? 5 < 6 -> sumOff-wins branch
        // cell 2: |D_internal| = 1 < sumOff (2*|3| = 6)        -> sumOff-wins branch
        // (off-diagonal count per cell depends on the CSR; computed from the matrix below.)
        std::vector<scalar> diagBase {-4.0, -5.0, -1.0};
        std::vector<scalar> offBase {1.0, 2.0, 3.0};
        std::vector<scalar> psiBase {0.0, 0.0, 0.0}; // psi=0 -> no source correction noise here
        auto [ls, solution] = buildSystem(diagBase, offBase, psiBase);

        // Capture the assembled CSR to hand-compute expected per cell.
        auto before = ls.copyToHost();
        auto beforeView = before.view();

        std::vector<ValueType> dAug(nCells), sumOff(nCells, NeoN::zero<ValueType>());
        for (localIdx c = 0; c < nCells; ++c)
        {
            const auto diagIdx = hostDiagIdx(beforeView.matrix, c);
            dAug[static_cast<std::size_t>(c)] = beforeView.matrix.values[diagIdx];
            for (localIdx idx = beforeView.matrix.sparsity.rowOffs[c];
                 idx < beforeView.matrix.sparsity.rowOffs[c + 1];
                 ++idx)
            {
                if (idx != diagIdx)
                {
                    sumOff[static_cast<std::size_t>(c)] =
                        sumOff[static_cast<std::size_t>(c)]
                        + hostCompMag(beforeView.matrix.values[idx]);
                }
            }
        }

        dsl::applyMatrixRelaxation(ls, solution, alpha);

        auto after = ls.copyToHost();
        auto afterView = after.view();
        for (localIdx c = 0; c < nCells; ++c)
        {
            // boundaryDiag == 0 here, so D_internal == D_aug.
            const ValueType dInternal = dAug[static_cast<std::size_t>(c)];
            const ValueType dDom =
                hostCompMax(hostCompMag(dInternal), sumOff[static_cast<std::size_t>(c)]);
            const ValueType expected = hostCopySign(invAlpha * dDom, dInternal);
            const ValueType got = afterView.matrix.values[hostDiagIdx(afterView.matrix, c)];
            INFO("clamp cell " << c);
            if constexpr (std::is_same_v<ValueType, scalar>)
            {
                REQUIRE(got == Catch::Approx(expected).margin(1e-12));
            }
            else
            {
                REQUIRE(got[0] == Catch::Approx(expected[0]).margin(1e-12));
                REQUIRE(got[1] == Catch::Approx(expected[1]).margin(1e-12));
                REQUIRE(got[2] == Catch::Approx(expected[2]).margin(1e-12));
            }
        }
    }

#if NF_WITH_GINKGO
    SECTION("fixedpoint invariance on " + execName)
    {
        // Build a diagonally dominant, negative-diagonal system. Choose a known x*,
        // set rhs = A_aug * x* so the UNRELAXED system has x* as its exact solution.
        // After relaxation the source correction (D_relaxed - D_aug) * x* is added, so
        // the relaxed system A' x = b' also has x* as fixed point. Solve and assert x*.
        std::vector<scalar> xStarBase {1.0, -2.0, 3.0};
        auto [ls, solution] =
            buildSystem({-4.0, -5.0, -6.0}, {1.0, 1.0, 1.0}, xStarBase); // dominant rows

        // psi_prev (solution.internalVector()) is x* -> the fixed point we test.
        // Compute rhs = A_aug * x* on host from the assembled CSR.
        auto sysH = ls.copyToHost();
        auto sysView = sysH.view();
        Vector<ValueType> xStar(exec, nCells);
        {
            auto xh = xStar.copyToHost();
            for (localIdx c = 0; c < nCells; ++c)
            {
                xh.view()[c] = sampleValue<ValueType>(xStarBase[static_cast<std::size_t>(c)]);
            }
            xStar = xh.copyToExecutor(exec);
        }
        auto xStarH = xStar.copyToHost();

        {
            auto rhsH = ls.rhs().copyToHost();
            for (localIdx c = 0; c < nCells; ++c)
            {
                ValueType acc = NeoN::zero<ValueType>();
                for (localIdx idx = sysView.matrix.sparsity.rowOffs[c];
                     idx < sysView.matrix.sparsity.rowOffs[c + 1];
                     ++idx)
                {
                    const localIdx col = sysView.matrix.sparsity.colIdxs[idx];
                    // componentwise product (matches the kernel's * semantics for Vec3).
                    acc = acc + sysView.matrix.values[idx] * xStarH.view()[col];
                }
                rhsH.view()[c] = acc;
            }
            ls.rhs() = rhsH.copyToExecutor(exec);
        }

        // Relax (alpha = 0.5). The source correction must keep x* stationary.
        dsl::applyMatrixRelaxation(ls, solution, 0.5);

        NeoN::Dictionary solverDict {
            {{"solver", std::string {"Ginkgo"}},
             {"type", "solver::Bicgstab"},
             {"criteria", NeoN::Dictionary {{{"iteration", 200}, {"relative_residual_norm", 1e-12}}}
             }}
        };
        auto solver = NeoN::la::Solver(exec, solverDict);

        Vector<ValueType> x(exec, nCells, NeoN::zero<ValueType>());
        solver.solve(ls, x);

        auto xH = x.copyToHost();
        for (localIdx c = 0; c < nCells; ++c)
        {
            INFO("fixedpoint cell " << c);
            if constexpr (std::is_same_v<ValueType, scalar>)
            {
                REQUIRE(xH.view()[c] == Catch::Approx(xStarH.view()[c]).margin(1e-9));
            }
            else
            {
                REQUIRE(xH.view()[c][0] == Catch::Approx(xStarH.view()[c][0]).margin(1e-9));
                REQUIRE(xH.view()[c][1] == Catch::Approx(xStarH.view()[c][1]).margin(1e-9));
                REQUIRE(xH.view()[c][2] == Catch::Approx(xStarH.view()[c][2]).margin(1e-9));
            }
        }
    }
#endif // NF_WITH_GINKGO
}

// ---------------------------------------------------------------------------
// Field (explicit) under-relaxation: dsl::applyFieldRelaxation +
// dsl::fieldRelaxationSnapshot.
//
// The kernel blends the INTERNAL vector only: psi = prev + alpha*(psi - prev),
// with an alpha<=0 || alpha==1 bitwise no-op early return. The snapshot helper returns
// an independent on-executor deep copy of the field's internal vector. Mirrors the
// applyMatrixRelaxation case above: TEMPLATE on scalar+Vec3, GENERATE over all executors,
// NeoN::fence before every host read (async-fence trap), independent host recompute as
// the oracle (never a second applyFieldRelaxation call), and bitwise (==, not Approx)
// no-op assertions.
// ---------------------------------------------------------------------------
TEMPLATE_TEST_CASE(
    "applyFieldRelaxation", "[relaxation][fieldrelax][snapshot][noop]", NeoN::scalar, NeoN::Vec3
)
{
    using ValueType = TestType;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx nCells = 3;
    auto mesh = NeoN::create1DUniformMesh(exec, nCells);

    // Companion boundary conditions (one per patch; only the internalVector is blended).
    std::vector<fvcc::VolumeBoundary<ValueType>> bcs {};
    for (localIdx patchi = 0; patchi < mesh.nBoundaries(); ++patchi)
    {
        NeoN::Dictionary dict;
        dict.insert("type", std::string("fixedValue"));
        dict.insert("fixedValue", NeoN::zero<ValueType>());
        bcs.push_back(fvcc::VolumeBoundary<ValueType>(mesh, dict, patchi));
    }

    // Build a VolumeField whose internal vector holds a known, asymmetric pattern so the
    // blend is observable per cell and per Vec3 component.
    auto makeField = [&](std::vector<scalar> base)
    {
        Vector<ValueType> psi(exec, nCells);
        {
            auto ph = psi.copyToHost();
            for (localIdx c = 0; c < nCells; ++c)
            {
                ph.view()[c] = sampleValue<ValueType>(base[static_cast<std::size_t>(c)]);
            }
            psi = ph.copyToExecutor(exec);
        }
        return fvcc::VolumeField<ValueType>(exec, "psi", mesh, psi, bcs);
    };

    SECTION("fieldRelax blend on " + execName)
    {
        const scalar alpha = 0.3;

        // prev = start-of-iteration state; snapshot it BEFORE mutating the field.
        auto solution = makeField({0.7, -0.3, 0.5});
        auto prev = dsl::fieldRelaxationSnapshot(solution);
        auto prevHostV = prev.copyToHost();

        // Mutate the field to a NEW "solved" state cur (add a known per-cell delta).
        std::vector<scalar> deltaBase {1.0, 2.0, -1.5};
        {
            auto cur = solution.internalVector().copyToHost();
            for (localIdx c = 0; c < nCells; ++c)
            {
                cur.view()[c] =
                    cur.view()[c] + sampleValue<ValueType>(deltaBase[static_cast<std::size_t>(c)]);
            }
            solution.internalVector() = cur.copyToExecutor(exec);
        }
        NeoN::fence(exec);

        // Capture cur (the pre-blend state) on host for the INDEPENDENT recompute oracle.
        auto curHostV = solution.internalVector().copyToHost();

        dsl::applyFieldRelaxation(solution, prev, alpha);
        NeoN::fence(exec);

        auto afterV = solution.internalVector().copyToHost();
        auto after = afterV.view();
        auto prevHost = prevHostV.view();
        auto curHost = curHostV.view();
        for (localIdx c = 0; c < nCells; ++c)
        {
            // Independent recompute: expected = prev + alpha*(cur - prev). NOT a second
            // applyFieldRelaxation call (independent-recompute oracle).
            const ValueType expected = prevHost[c] + alpha * (curHost[c] - prevHost[c]);
            INFO("fieldRelax cell " << c);
            if constexpr (std::is_same_v<ValueType, scalar>)
            {
                REQUIRE(after[c] == Catch::Approx(expected).margin(1e-12));
            }
            else
            {
                REQUIRE(after[c][0] == Catch::Approx(expected[0]).margin(1e-12));
                REQUIRE(after[c][1] == Catch::Approx(expected[1]).margin(1e-12));
                REQUIRE(after[c][2] == Catch::Approx(expected[2]).margin(1e-12));
            }
        }
    }

    SECTION("fieldRelaxSnapshot is an independent deep copy on " + execName)
    {
        auto solution = makeField({0.7, -0.3, 0.5});

        auto prev = dsl::fieldRelaxationSnapshot(solution);
        REQUIRE(prev.size() == solution.size());

        auto prevHost0V = prev.copyToHost();
        std::vector<ValueType> prevHost0(static_cast<std::size_t>(nCells));
        for (localIdx c = 0; c < nCells; ++c)
        {
            prevHost0[static_cast<std::size_t>(c)] = prevHost0V.view()[c];
        }

        // Mutate the SOURCE field to a completely different state; the snapshot must NOT track it.
        {
            auto cur = solution.internalVector().copyToHost();
            for (localIdx c = 0; c < nCells; ++c)
            {
                cur.view()[c] = sampleValue<ValueType>(-9.0 * (static_cast<scalar>(c) + 1.0));
            }
            solution.internalVector() = cur.copyToExecutor(exec);
        }
        NeoN::fence(exec);

        auto prevHost1V = prev.copyToHost();
        auto prevHost1 = prevHost1V.view();
        for (localIdx c = 0; c < nCells; ++c)
        {
            const ValueType expect = prevHost0[static_cast<std::size_t>(c)];
            INFO("snapshot cell " << c);
            // BITWISE (==, not Approx): a deep copy is byte-identical and independent.
            if constexpr (std::is_same_v<ValueType, scalar>)
            {
                REQUIRE(prevHost1[c] == expect);
            }
            else
            {
                REQUIRE(prevHost1[c][0] == expect[0]);
                REQUIRE(prevHost1[c][1] == expect[1]);
                REQUIRE(prevHost1[c][2] == expect[2]);
            }
        }
    }

    SECTION("alpha==1 / alpha<=0 is a bitwise no-op on " + execName)
    {
        auto solution = makeField({0.7, -0.3, 0.5});

        // Capture the field state BEFORE any relaxation.
        auto beforeV = solution.internalVector().copyToHost();
        std::vector<ValueType> before(static_cast<std::size_t>(nCells));
        for (localIdx c = 0; c < nCells; ++c)
        {
            before[static_cast<std::size_t>(c)] = beforeV.view()[c];
        }

        // Build a prev whose values DIFFER from the current field (prev != current is the
        // point: prev + 1*(cur-prev) need not round-trip to cur bitwise; the
        // early return must side-step that entirely).
        Vector<ValueType> prev(exec, nCells);
        {
            auto ph = prev.copyToHost();
            for (localIdx c = 0; c < nCells; ++c)
            {
                ph.view()[c] = before[static_cast<std::size_t>(c)]
                             + sampleValue<ValueType>(0.123456789 * (static_cast<scalar>(c) + 1.0));
            }
            prev = ph.copyToExecutor(exec);
        }

        auto assertUnchanged = [&](scalar alpha)
        {
            dsl::applyFieldRelaxation(solution, prev, alpha);
            NeoN::fence(exec);
            auto afterV = solution.internalVector().copyToHost();
            auto after = afterV.view();
            for (localIdx c = 0; c < nCells; ++c)
            {
                const ValueType b = before[static_cast<std::size_t>(c)];
                INFO("noop alpha=" << alpha << " cell " << c);
                if constexpr (std::is_same_v<ValueType, scalar>)
                {
                    REQUIRE(after[c] == b);
                }
                else
                {
                    REQUIRE(after[c][0] == b[0]);
                    REQUIRE(after[c][1] == b[1]);
                    REQUIRE(after[c][2] == b[2]);
                }
            }
        };

        assertUnchanged(1.0);  // final outer iteration / unset factor
        assertUnchanged(0.0);  // alpha == 0 guard
        assertUnchanged(-0.5); // alpha < 0 guard
    }
}

// ---------------------------------------------------------------------------
// CSR-vs-ELL parity for applyMatrixRelaxation, on a real assembled Laplacian -- unlike the
// TEMPLATE_TEST_CASE above (whose job is proving CSR's own relaxation math on a hand-built
// synthetic matrix), this one's job is proving the format-generic rewrite of the row-sum kernel
// (SparsityView/EllSparsityView rowSize()/linearIndex(), replacing the old raw rowOffs()/colIdxs()
// walk that never compiled for ELL) produces identical results on both formats. Nonuniform gamma
// so off-diagonal magnitudes differ per cell, keeping the dominance-clamp path meaningful.
// ---------------------------------------------------------------------------
TEST_CASE("applyMatrixRelaxation matches for CSR and ELL")
{
    using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = NeoN::create2DUniformMesh(exec, 4, 4);
    auto nCells = mesh.nCells();

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    const auto nInternalFaces = mesh.nInternalFaces();
    auto gammaV = gamma.internalVector().view();
    NeoN::parallelFor(
        exec,
        {0, nInternalFaces},
        NEON_LAMBDA(const localIdx facei) {
            gammaV[facei] = 1.0 + 0.1 * static_cast<scalar>(facei);
        }
    );
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    NeoN::Input faceNormalGradientInput = NeoN::TokenList({std::string("uncorrected")});
    fvcc::FaceNormalGradient<scalar> faceNormalGradient(exec, mesh, faceNormalGradientInput);

    auto csrLs = NeoN::la::createEmptyLinearSystem<scalar, scalar, CSRMatrix>(mesh);
    auto ellLs = NeoN::la::createEmptyLinearSystem<scalar, scalar, ELLMatrix>(mesh);
    fvcc::computeLaplacianIntImpl(csrLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);
    fvcc::computeLaplacianIntImpl(ellLs, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);

    // Nonzero rhs so a regressed guard touching rhs is caught.
    fill(csrLs.rhs(), 1.0);
    fill(ellLs.rhs(), 1.0);

    dsl::applyMatrixRelaxation(csrLs, phi, 0.5);
    dsl::applyMatrixRelaxation(ellLs, phi, 0.5);

    REQUIRE_THAT(csrLs.matrix().diag(), Equals(ellLs.matrix().diag(), Approx {1e-10}));
    REQUIRE_THAT(csrLs.rhs(), Equals(ellLs.rhs(), Approx {1e-10}));

    // Compare every logical (row,col) entry -- not the flat values() arrays, which have
    // different physical layouts (CSR compact vs ELL padded column-major).
    auto csrLsHost = csrLs.copyToExecutor(NeoN::SerialExecutor());
    auto ellLsHost = ellLs.copyToExecutor(NeoN::SerialExecutor());
    auto csrSparsityView = csrLsHost.matrix().sparsity()->view();
    auto csrMatView = csrLsHost.matrix().view();
    auto ellMatView = ellLsHost.matrix().view();

    std::vector<scalar> csrEntries;
    std::vector<scalar> ellEntries;
    for (localIdx row = 0; row < nCells; ++row)
    {
        for (localIdx col = 0; col < nCells; ++col)
        {
            if (csrSparsityView.findEntry(row, col) != decltype(csrSparsityView)::invalidIndex())
            {
                csrEntries.push_back(csrMatView.entry(row, col));
                ellEntries.push_back(ellMatView.entry(row, col));
            }
        }
    }
    REQUIRE(csrEntries.size() == ellEntries.size());
    REQUIRE_THAT(
        Vector<scalar>(NeoN::SerialExecutor(), ellEntries), Equals(csrEntries, Approx {1e-10})
    );

    // Every ELL slot whose column index is the padding sentinel must stay untouched.
    auto colIdxHostV = ellLsHost.matrix().sparsity()->colIdxs().view();
    auto ellValuesHostV = ellLsHost.matrix().values().view();
    for (localIdx i = 0; i < colIdxHostV.size(); ++i)
    {
        if (colIdxHostV[i] == decltype(ellLsHost.matrix().sparsity()->view())::invalidIndex())
        {
            REQUIRE(ellValuesHostV[i] == NeoN::zero<scalar>());
        }
    }
}

// alpha == 1 / alpha <= 0 bitwise no-op, on ELL specifically. The existing TEMPLATE_TEST_CASE
// above proves this for CSR; the guard is the very first thing the function does, before any
// format-specific addressing, but the row-walk rewrite touched every line after it, so this
// confirms directly on ELL rather than by inference from CSR.
TEST_CASE("applyMatrixRelaxation alpha==1/alpha<=0 is a bitwise no-op on ELL")
{
    using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = NeoN::create2DUniformMesh(exec, 4, 4);

    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);
    fvcc::SurfaceField<scalar> gamma(exec, "gamma", mesh, surfaceBCs);
    fill(gamma.internalVector(), 1.0);
    fill(gamma.boundaryData().value(), 1.0);

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(mesh);
    fvcc::VolumeField<scalar> phi(exec, "phi", mesh, volumeBCs);
    Catch::randomizeVector(phi);
    phi.correctBoundaryConditions();

    NeoN::Input faceNormalGradientInput = NeoN::TokenList({std::string("uncorrected")});
    fvcc::FaceNormalGradient<scalar> faceNormalGradient(exec, mesh, faceNormalGradientInput);

    auto buildEllSystem = [&]()
    {
        auto ls = NeoN::la::createEmptyLinearSystem<scalar, scalar, ELLMatrix>(mesh);
        fvcc::computeLaplacianIntImpl(ls, gamma, phi, dsl::Coeff {1.0}, faceNormalGradient);
        fill(ls.rhs(), 1.0);
        return ls;
    };

    for (scalar alpha : {1.0, 0.0, -0.5})
    {
        auto ls = buildEllSystem();
        auto lsRef = buildEllSystem();
        dsl::applyMatrixRelaxation(ls, phi, alpha);
        INFO("alpha=" << alpha);

        auto values = ls.matrix().values().copyToHost();
        auto refValues = lsRef.matrix().values().copyToHost();
        auto rhs = ls.rhs().copyToHost();
        auto refRhs = lsRef.rhs().copyToHost();

        for (localIdx i = 0; i < values.size(); ++i)
        {
            REQUIRE(values.view()[i] == refValues.view()[i]);
        }
        for (localIdx i = 0; i < rhs.size(); ++i)
        {
            REQUIRE(rhs.view()[i] == refRhs.view()[i]);
        }
    }
}
