// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

#if NF_WITH_GINKGO

#include <filesystem>
#include <fstream>
#include <string>

using NeoN::Dictionary;
using NeoN::localIdx;
using NeoN::scalar;
using NeoN::Vector;
using NeoN::la::COOMatrix;
using NeoN::la::CooSparsityPattern;
using NeoN::la::CSRMatrix;
using NeoN::la::CsrSparsityPattern;
using NeoN::la::Dimensions;
using NeoN::la::LinearSystem;

namespace
{

/**
 * @brief End-to-end coverage of the Ginkgo multigrid `scale_correction` parameter.
 *
 * `scale_correction` is the OpenFOAM-style per-level Rayleigh-quotient scaling of the pre-smooth
 * and prolonged-coarse corrections (GAMGSolverSolve.C + GAMGSolverScale.C). It lives inside
 * Ginkgo's `MultigridState::run_cycle` and reaches NeoN through the solver dictionary's
 * `configFile` entry, so these tests drive the whole chain: NeoN Dictionary -> ginkgo::parse ->
 * gko::config::parse -> Multigrid::parse -> run_cycle.
 */

/** @brief 1D Laplacian tridiag(-1, 2, -1): SPD, and large enough for Pgm to build several levels
 * (scale correction is only active on levels above the coarsest). */
struct Laplacian1D
{
    localIdx n;
    std::shared_ptr<CsrSparsityPattern<localIdx>> sparsity;
    std::shared_ptr<CooSparsityPattern<localIdx>> bSparsity;
    Vector<scalar> values;

    Laplacian1D(const NeoN::Executor& exec, localIdx nRows) : n(nRows), values(exec, localIdx {0})
    {
        std::vector<localIdx> cols;
        std::vector<localIdx> rows {0};
        std::vector<scalar> vals;
        for (localIdx i = 0; i < n; ++i)
        {
            if (i > 0)
            {
                cols.push_back(i - 1);
                vals.push_back(-1.0);
            }
            cols.push_back(i);
            vals.push_back(2.0);
            if (i < n - 1)
            {
                cols.push_back(i + 1);
                vals.push_back(-1.0);
            }
            rows.push_back(static_cast<localIdx>(cols.size()));
        }

        sparsity = std::make_shared<CsrSparsityPattern<localIdx>>(
            Vector<localIdx>(exec, cols), Vector<localIdx>(exec, rows), Dimensions {n, n}
        );
        bSparsity = std::make_shared<CooSparsityPattern<localIdx>>(
            Vector<localIdx>(exec, {}), Vector<localIdx>(exec, {}), Dimensions {0, 0}
        );
        values = Vector<scalar>(exec, vals);
    }

    /** @brief System with rhs = A * ones, so the exact solution is the all-ones vector. */
    LinearSystem<scalar> system(const NeoN::Executor& exec) const
    {
        CSRMatrix<scalar, localIdx> mtx(values, sparsity);

        std::vector<scalar> rhsHost(static_cast<size_t>(n), 0.0);
        for (localIdx i = 0; i < n; ++i)
        {
            // (A * 1)_i = 2 - (i > 0) - (i < n-1)
            rhsHost[static_cast<size_t>(i)] = 2.0 - (i > 0 ? 1.0 : 0.0) - (i < n - 1 ? 1.0 : 0.0);
        }

        Vector<scalar> bValues(exec, {});
        COOMatrix<scalar, localIdx> bMtx(bValues, bSparsity);
        Vector<scalar> bRhs(exec, {});
        return LinearSystem<scalar>(mtx, Vector<scalar>(exec, rhsHost), bMtx, bMtx, bRhs);
    }
};

/** @brief Ir(2 sweeps) + point Jacobi, used as pre- and post-smoother on every level. */
std::string smootherJson()
{
    return R"({"type": "solver::Ir",
               "solver": {"type": "preconditioner::Jacobi", "max_block_size": 1},
               "relaxation_factor": 0.6666666666666666,
               "criteria": [{"type": "Iteration", "max_iters": 2}]})";
}

std::string coarsestJson()
{
    return R"({"type": "solver::Cg",
               "criteria": [{"type": "Iteration", "max_iters": 50},
                            {"type": "ResidualNorm", "reduction_factor": 1e-4}]})";
}

/** @brief A Pgm V-cycle multigrid node. `criteria` bounds the number of cycles; when used as a
 * preconditioner that is a single cycle. */
std::string multigridJson(bool scaleCorrection, const std::string& criteria, bool providedGuess)
{
    std::string s = R"({"type": "solver::Multigrid",
        "mg_level": [{"type": "multigrid::Pgm", "deterministic": true}],
        "pre_smoother": [)"
                  + smootherJson() + R"(],
        "post_smoother": [)"
                  + smootherJson() + R"(],
        "coarsest_solver": [)"
                  + coarsestJson() + R"(],
        "min_coarse_rows": 16,
        "cycle": "v",
        "scale_correction": )"
                  + (scaleCorrection ? "true" : "false");
    if (providedGuess)
    {
        s += R"(, "default_initial_guess": "provided")";
    }
    s += R"(, "criteria": [)" + criteria + "]}";
    return s;
}

/** @brief CG preconditioned by one (possibly scale-corrected) multigrid V-cycle. This is how the
 * OpenFOAM-style correction is deployed: it is nonlinear, so an outer Krylov drives it. */
std::string cgMgJson(bool scaleCorrection)
{
    return R"({"type": "solver::Cg",
        "criteria": [{"type": "Iteration", "max_iters": 300},
                     {"type": "ResidualNorm", "baseline": "rhs_norm",
                      "reduction_factor": 1e-11}],
        "preconditioner": )"
         + multigridJson(scaleCorrection, R"({"type": "Iteration", "max_iters": 1})", false) + "}";
}

/** @brief Writes @p json to a uniquely named file and returns the path; Ginkgo's config reader
 * takes a filename, which is also the path NeoN production configs use. */
std::string writeConfig(const std::string& name, const std::string& json)
{
    auto path = std::filesystem::temp_directory_path() / ("neon_sc_" + name + ".json");
    std::ofstream out(path);
    out << json;
    out.close();
    return path.string();
}

Dictionary solverDict(const std::string& configPath)
{
    return Dictionary {{{"solver", std::string {"Ginkgo"}}, {"configFile", configPath}}};
}

std::vector<scalar> solveWith(
    const NeoN::Executor& exec,
    const LinearSystem<scalar>& sys,
    const std::string& configPath,
    localIdx n
)
{
    auto solver = NeoN::la::Solver(exec, solverDict(configPath));
    Vector<scalar> x(exec, n, 0.0);
    solver.solve(sys, x);
    auto host = x.copyToHost();
    auto view = host.view();
    return std::vector<scalar>(view.data(), view.data() + n);
}

scalar maxAbsDiff(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

} // namespace


TEST_CASE("Multigrid scale_correction - Ginkgo")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    const localIdx n = 300;
    Laplacian1D lap(exec, n);
    auto sys = lap.system(exec);

    SECTION("scale-corrected V-cycle preconditions CG to the exact solution " + execName)
    {
        // Drives the whole scale-correction path (pre-smooth + coarse Rayleigh scaling through
        // the device-side safe_inv_scale) once per outer CG iteration.
        auto path = writeConfig("cg_mg_sc", cgMgJson(true));
        auto x = solveWith(exec, sys, path, n);

        for (localIdx i = 0; i < n; ++i)
        {
            REQUIRE(x[static_cast<size_t>(i)] == Catch::Approx(1.0).margin(1e-8));
        }
    }

    SECTION("scale correction does not move the fixed point " + execName)
    {
        // Scale correction changes the multigrid convergence path but not what it converges to:
        // both configurations must drive CG to the same solution.
        auto xOff = solveWith(exec, sys, writeConfig("cg_mg_off", cgMgJson(false)), n);
        auto xOn = solveWith(exec, sys, writeConfig("cg_mg_on", cgMgJson(true)), n);

        REQUIRE(maxAbsDiff(xOff, xOn) < 1e-6);
    }

    SECTION("the flag is honoured, not silently ignored " + execName)
    {
        // Guards against scale_correction being dropped on the floor by the config layer (a
        // wrong key name, or an unpatched Ginkgo whose Multigrid::parse has no such hook). Three
        // fixed V-cycles with an iteration-only criterion: if the flag reached run_cycle the
        // iterate differs, if it was ignored the two runs are bit-identical.
        const std::string cycles = R"({"type": "Iteration", "max_iters": 3})";
        auto xOff = solveWith(
            exec, sys, writeConfig("mg_fixed_off", multigridJson(false, cycles, true)), n
        );
        auto xOn =
            solveWith(exec, sys, writeConfig("mg_fixed_on", multigridJson(true, cycles, true)), n);

        REQUIRE(maxAbsDiff(xOff, xOn) > 1e-10);
        for (localIdx i = 0; i < n; ++i)
        {
            REQUIRE(std::isfinite(xOn[static_cast<size_t>(i)]));
        }
    }

    SECTION("zero rhs stays zero without NaN " + execName)
    {
        // Regression guard for the device-side reciprocal in ginkgo_local_stack.patch: with a
        // zero rhs and zero initial guess every correction delta is exactly zero, so the Rayleigh
        // denominator delta.A.delta is exactly zero. The guard must give a scale factor of
        // 0/(0+eps) = 0, not a 0/0 NaN.
        CSRMatrix<scalar, localIdx> mtx(lap.values, lap.sparsity);
        Vector<scalar> bValues(exec, {});
        COOMatrix<scalar, localIdx> bMtx(bValues, lap.bSparsity);
        Vector<scalar> bRhs(exec, {});
        auto zeroSys = LinearSystem<scalar>(mtx, Vector<scalar>(exec, n, 0.0), bMtx, bMtx, bRhs);

        const std::string cycles = R"({"type": "Iteration", "max_iters": 3})";
        auto x = solveWith(
            exec, zeroSys, writeConfig("mg_zero_rhs", multigridJson(true, cycles, true)), n
        );

        for (localIdx i = 0; i < n; ++i)
        {
            REQUIRE(std::isfinite(x[static_cast<size_t>(i)]));
            REQUIRE(x[static_cast<size_t>(i)] == Catch::Approx(0.0).margin(1e-14));
        }
    }
}

#endif
