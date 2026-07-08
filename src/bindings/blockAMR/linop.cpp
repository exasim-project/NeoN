// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

#include <AMReX_Config.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MLMG.H>

#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLEBTensorOp.H>
#include <AMReX_EBFabFactory.H>
#endif

namespace nb = nanobind;

void registerLinOp(nb::module_& m)
{
    using namespace amrex;

    using MLLinOp = MLLinOpT<MultiFab>;
    using MLPoisson = MLPoissonT<MultiFab>;
    using MLABecLap = MLABecLaplacianT<MultiFab>;
    using MLMG = MLMGT<MultiFab>;

    // --- LinOpBCType enum ---
    nb::enum_<LinOpBCType>(m, "LinOpBCType")
        .value("interior", LinOpBCType::interior)
        .value("Dirichlet", LinOpBCType::Dirichlet)
        .value("Neumann", LinOpBCType::Neumann)
        .value("reflect_odd", LinOpBCType::reflect_odd)
        .value("Marshak", LinOpBCType::Marshak)
        .value("SanchezPomraning", LinOpBCType::SanchezPomraning)
        .value("inflow", LinOpBCType::inflow)
        .value("inhomogNeumann", LinOpBCType::inhomogNeumann)
        .value("Robin", LinOpBCType::Robin)
        .value("symmetry", LinOpBCType::symmetry)
        .value("Periodic", LinOpBCType::Periodic)
        .value("bogus", LinOpBCType::bogus);

    // --- LPInfo ---
    nb::class_<LPInfo>(m, "LPInfo")
        .def(nb::init<>())
        .def(
            "set_max_coarsening_level",
            [](LPInfo& info, int n) -> LPInfo& { return info.setMaxCoarseningLevel(n); },
            nb::arg("n"),
            nb::rv_policy::reference
        )
        .def(
            "set_agglomeration",
            [](LPInfo& info, bool x) -> LPInfo& { return info.setAgglomeration(x); },
            nb::arg("x"),
            nb::rv_policy::reference
        )
        .def(
            "set_consolidation",
            [](LPInfo& info, bool x) -> LPInfo& { return info.setConsolidation(x); },
            nb::arg("x"),
            nb::rv_policy::reference
        );

    // --- Base class (opaque, needed for MLMG to accept derived types) ---
    nb::class_<MLLinOp>(m, "MLLinOp")
        .def(
            "fix_solvability",
            // Compute the AMReX-internal solvability offset for the
            // singular all-Neumann case and apply it to the rhs in
            // place. Wraps ``getSolvabilityOffset`` +
            // ``fixSolvabilityByOffset``. For MLNodeLaplacian + EB
            // this is a volume-fraction-weighted projection that the
            // automatic ``MLMG::makeSolvable`` skips because
            // ``isSingular()`` returns false on an EB factory (see
            // ``MLNodeLinOp.cpp:295``: m_is_bottom_singular gates on
            // m_domain_covered which is false with EB).
            [](MLLinOp& lp, MultiFab& rhs)
            {
                auto offset = lp.getSolvabilityOffset(0, 0, rhs);
                lp.fixSolvabilityByOffset(0, 0, rhs, offset);
            },
            nb::arg("rhs")
        );

    // Helper: convert Python array of LinOpBCType to AMReX Array
    auto setDomainBC = [](MLLinOp& lp,
                          std::array<LinOpBCType, AMREX_SPACEDIM> lo_bc,
                          std::array<LinOpBCType, AMREX_SPACEDIM> hi_bc)
    {
        Array<LinOpBCType, AMREX_SPACEDIM> lo, hi;
        for (int i = 0; i < AMREX_SPACEDIM; ++i)
        {
            lo[i] = lo_bc[static_cast<std::size_t>(i)];
            hi[i] = hi_bc[static_cast<std::size_t>(i)];
        }
        lp.setDomainBC(lo, hi);
    };

    // --- MLPoisson: del dot grad phi ---
    nb::class_<MLPoisson, MLLinOp>(m, "MLPoisson")
        .def(
            "__init__",
            [](MLPoisson* self,
               const Geometry& geom,
               const BoxArray& ba,
               const DistributionMapping& dm,
               const LPInfo& info)
            { new (self) MLPoisson({geom}, {ba}, {dm}, info); },
            nb::arg("geom"),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("info") = LPInfo()
        )
        .def("set_domain_bc", setDomainBC, nb::arg("lo_bc"), nb::arg("hi_bc"))
        .def(
            "set_level_bc",
            [](MLPoisson& lp, int lev, MultiFab* levdata) { lp.setLevelBC(lev, levdata); },
            nb::arg("lev"),
            nb::arg("levdata") = nullptr
        );

    // --- MLABecLaplacian: (alpha * a - beta * div(b * grad)) phi ---
    nb::class_<MLABecLap, MLLinOp>(m, "MLABecLaplacian")
        .def(
            "__init__",
            [](MLABecLap* self,
               const Geometry& geom,
               const BoxArray& ba,
               const DistributionMapping& dm,
               const LPInfo& info)
            { new (self) MLABecLap({geom}, {ba}, {dm}, info); },
            nb::arg("geom"),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("info") = LPInfo()
        )
        .def("set_domain_bc", setDomainBC, nb::arg("lo_bc"), nb::arg("hi_bc"))
        .def(
            "set_level_bc",
            [](MLABecLap& lp, int lev, MultiFab* levdata) { lp.setLevelBC(lev, levdata); },
            nb::arg("lev"),
            nb::arg("levdata") = nullptr
        )
        .def(
            "set_scalars",
            [](MLABecLap& lp, double a, double b) { lp.setScalars(a, b); },
            nb::arg("a"),
            nb::arg("b")
        )
        .def(
            "set_a_coeffs",
            [](MLABecLap& lp, int lev, const MultiFab& alpha) { lp.setACoeffs(lev, alpha); },
            nb::arg("lev"),
            nb::arg("alpha")
        )
        .def(
            "set_b_coeffs",
            [](MLABecLap& lp, int lev, const MultiFab& bx, const MultiFab& by, const MultiFab& bz)
            {
                Array<MultiFab const*, AMREX_SPACEDIM> beta = {
                    AMREX_D_DECL(&bx, &by, &bz)};
                lp.setBCoeffs(lev, beta);
            },
            nb::arg("lev"),
            nb::arg("bx"),
            nb::arg("by"),
            nb::arg("bz")
        );

#ifdef AMREX_USE_EB
    // --- MLEBABecLaplacian: same equation as MLABecLap but EB-aware ---
    nb::class_<MLEBABecLap, MLLinOp>(m, "MLEBABecLaplacian")
        .def(
            "__init__",
            [](MLEBABecLap* self,
               const Geometry& geom,
               const BoxArray& ba,
               const DistributionMapping& dm,
               const LPInfo& info,
               const EBFArrayBoxFactory& factory)
            {
                Vector<EBFArrayBoxFactory const*> facs{&factory};
                new (self) MLEBABecLap({geom}, {ba}, {dm}, info, facs);
            },
            nb::arg("geom"),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("info"),
            nb::arg("factory"),
            nb::keep_alive<1, 6>()
        )
        .def("set_domain_bc", setDomainBC, nb::arg("lo_bc"), nb::arg("hi_bc"))
        .def(
            "set_level_bc",
            [](MLEBABecLap& lp, int lev, MultiFab* levdata) { lp.setLevelBC(lev, levdata); },
            nb::arg("lev"),
            nb::arg("levdata") = nullptr
        )
        .def(
            "set_scalars",
            [](MLEBABecLap& lp, double a, double b) { lp.setScalars(a, b); },
            nb::arg("a"),
            nb::arg("b")
        )
        .def(
            "set_a_coeffs",
            [](MLEBABecLap& lp, int lev, const MultiFab& alpha) { lp.setACoeffs(lev, alpha); },
            nb::arg("lev"),
            nb::arg("alpha")
        )
        .def(
            "set_a_coeffs",
            [](MLEBABecLap& lp, int lev, Real alpha) { lp.setACoeffs(lev, alpha); },
            nb::arg("lev"),
            nb::arg("alpha")
        )
        .def(
            "set_b_coeffs",
            [](MLEBABecLap& lp, int lev, const MultiFab& bx, const MultiFab& by, const MultiFab& bz)
            {
                Array<MultiFab const*, AMREX_SPACEDIM> beta = {AMREX_D_DECL(&bx, &by, &bz)};
                lp.setBCoeffs(lev, beta);
            },
            nb::arg("lev"),
            nb::arg("bx"),
            nb::arg("by"),
            nb::arg("bz")
        )
        .def(
            "set_b_coeffs",
            [](MLEBABecLap& lp, int lev, Real beta) { lp.setBCoeffs(lev, beta); },
            nb::arg("lev"),
            nb::arg("beta")
        )
        .def(
            "set_eb_homog_dirichlet",
            [](MLEBABecLap& lp, int lev, Real beta) { lp.setEBHomogDirichlet(lev, beta); },
            nb::arg("lev"),
            nb::arg("beta")
        )
        .def(
            "set_eb_homog_dirichlet",
            [](MLEBABecLap& lp, int lev, const MultiFab& beta)
            { lp.setEBHomogDirichlet(lev, beta); },
            nb::arg("lev"),
            nb::arg("beta")
        )
        .def(
            "set_eb_dirichlet",
            [](MLEBABecLap& lp, int lev, const MultiFab& phi, Real beta)
            { lp.setEBDirichlet(lev, phi, beta); },
            nb::arg("lev"),
            nb::arg("phi"),
            nb::arg("beta")
        )
        .def(
            "set_eb_dirichlet",
            [](MLEBABecLap& lp, int lev, const MultiFab& phi, const MultiFab& beta)
            { lp.setEBDirichlet(lev, phi, beta); },
            nb::arg("lev"),
            nb::arg("phi"),
            nb::arg("beta")
        );

    // --- MLEBTensorOp: viscous-stress tensor operator with EB ---
    // Inherits from MLEBABecLap, so it slots in under MLLinOp via MLEBABecLap
    // (multi-level inheritance is fine for nanobind dispatch).
    nb::class_<MLEBTensorOp, MLEBABecLap>(m, "MLEBTensorOp")
        .def(
            "__init__",
            [](MLEBTensorOp* self,
               const Geometry& geom,
               const BoxArray& ba,
               const DistributionMapping& dm,
               const LPInfo& info,
               const EBFArrayBoxFactory& factory)
            {
                Vector<EBFArrayBoxFactory const*> facs{&factory};
                new (self) MLEBTensorOp({geom}, {ba}, {dm}, info, facs);
            },
            nb::arg("geom"),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("info"),
            nb::arg("factory"),
            nb::keep_alive<1, 6>()
        )
        .def(
            "set_shear_viscosity",
            [](MLEBTensorOp& lp, int lev, Real eta) { lp.setShearViscosity(lev, eta); },
            nb::arg("lev"),
            nb::arg("eta")
        )
        .def(
            "set_shear_viscosity",
            [](MLEBTensorOp& lp,
               int lev,
               const MultiFab& ex,
               const MultiFab& ey,
               const MultiFab& ez,
               int loc)
            {
                Array<MultiFab const*, AMREX_SPACEDIM> eta{AMREX_D_DECL(&ex, &ey, &ez)};
                lp.setShearViscosity(lev, eta, static_cast<MLLinOp::Location>(loc));
            },
            nb::arg("lev"),
            nb::arg("ex"),
            nb::arg("ey"),
            nb::arg("ez"),
            nb::arg("loc") = static_cast<int>(MLLinOp::Location::FaceCenter)
        )
        .def(
            "set_bulk_viscosity",
            [](MLEBTensorOp& lp, int lev, Real kappa) { lp.setBulkViscosity(lev, kappa); },
            nb::arg("lev"),
            nb::arg("kappa")
        )
        .def(
            "set_eb_shear_viscosity",
            [](MLEBTensorOp& lp, int lev, Real eta) { lp.setEBShearViscosity(lev, eta); },
            nb::arg("lev"),
            nb::arg("eta")
        )
        .def(
            "set_eb_bulk_viscosity",
            [](MLEBTensorOp& lp, int lev, Real kappa) { lp.setEBBulkViscosity(lev, kappa); },
            nb::arg("lev"),
            nb::arg("kappa")
        );
#endif

    // --- MLNodeLaplacian: del dot (sigma * grad phi) at nodes ---
    nb::class_<MLNodeLaplacian, MLLinOp>(m, "MLNodeLaplacian")
        .def(
            "__init__",
            [](MLNodeLaplacian* self,
               const Geometry& geom,
               const BoxArray& ba,
               const DistributionMapping& dm,
               const LPInfo& info,
               Real const_sigma)
            {
                Vector<FabFactory<FArrayBox> const*> factory;
                new (self) MLNodeLaplacian({geom}, {ba}, {dm}, info, factory, const_sigma);
            },
            nb::arg("geom"),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("info") = LPInfo(),
            nb::arg("const_sigma") = Real(0.0)
        )
        .def(
            "__init__",
            [](MLNodeLaplacian* self,
               nb::list geoms_py, nb::list bas_py, nb::list dms_py,
               const LPInfo& info,
               Real const_sigma)
            {
                auto n = nb::len(geoms_py);
                Vector<Geometry> geoms;
                Vector<BoxArray> bas;
                Vector<DistributionMapping> dms;
                geoms.reserve(n);
                bas.reserve(n);
                dms.reserve(n);
                for (size_t i = 0; i < n; ++i)
                {
                    geoms.push_back(nb::cast<Geometry>(geoms_py[i]));
                    bas.push_back(nb::cast<BoxArray>(bas_py[i]));
                    dms.push_back(nb::cast<DistributionMapping>(dms_py[i]));
                }
                Vector<FabFactory<FArrayBox> const*> factory;
                new (self) MLNodeLaplacian(geoms, bas, dms, info, factory, const_sigma);
            },
            nb::arg("geoms"),
            nb::arg("bas"),
            nb::arg("dms"),
            nb::arg("info") = LPInfo(),
            nb::arg("const_sigma") = Real(0.0)
        )
#ifdef AMREX_USE_EB
        // --- EB-aware constructors. Single level. ---
        // Same MLNodeLaplacian class, just constructed with an EB factory
        // so the operator knows about volume / area fractions internally.
        // This is the path IAMReX uses (their NavierStokesBase wires
        // EBFArrayBoxFactory through to MLNodeLaplacian via amrex-hydro's
        // NodalProjector, which is itself only a thin wrapper around the
        // same constructor we expose here).
        .def(
            "__init__",
            [](MLNodeLaplacian* self,
               const Geometry& geom,
               const BoxArray& ba,
               const DistributionMapping& dm,
               const LPInfo& info,
               const EBFArrayBoxFactory& factory,
               Real const_sigma)
            {
                Vector<EBFArrayBoxFactory const*> facs{&factory};
                new (self) MLNodeLaplacian({geom}, {ba}, {dm}, info, facs, const_sigma);
            },
            nb::arg("geom"),
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("info"),
            nb::arg("factory"),
            nb::arg("const_sigma") = Real(0.0),
            nb::keep_alive<1, 6>()
        )
        .def(
            "__init__",
            [](MLNodeLaplacian* self,
               nb::list geoms_py, nb::list bas_py, nb::list dms_py,
               const LPInfo& info,
               nb::list factories_py,
               Real const_sigma)
            {
                auto n = nb::len(geoms_py);
                Vector<Geometry> geoms;
                Vector<BoxArray> bas;
                Vector<DistributionMapping> dms;
                Vector<EBFArrayBoxFactory const*> facs;
                geoms.reserve(n);
                bas.reserve(n);
                dms.reserve(n);
                facs.reserve(n);
                for (size_t i = 0; i < n; ++i)
                {
                    geoms.push_back(nb::cast<Geometry>(geoms_py[i]));
                    bas.push_back(nb::cast<BoxArray>(bas_py[i]));
                    dms.push_back(nb::cast<DistributionMapping>(dms_py[i]));
                    facs.push_back(&nb::cast<EBFArrayBoxFactory const&>(factories_py[i]));
                }
                new (self) MLNodeLaplacian(geoms, bas, dms, info, facs, const_sigma);
            },
            nb::arg("geoms"),
            nb::arg("bas"),
            nb::arg("dms"),
            nb::arg("info"),
            nb::arg("factories"),
            nb::arg("const_sigma") = Real(0.0),
            nb::keep_alive<1, 6>()
        )
#endif
        .def("set_domain_bc", setDomainBC, nb::arg("lo_bc"), nb::arg("hi_bc"))
        .def(
            "set_sigma",
            [](MLNodeLaplacian& lp, int lev, const MultiFab& sigma)
            { lp.setSigma(lev, sigma); },
            nb::arg("lev"),
            nb::arg("sigma")
        )
        .def(
            "comp_divergence",
            [](MLNodeLaplacian& lp, MultiFab& rhs, MultiFab& vel)
            { lp.compDivergence({&rhs}, {&vel}); },
            nb::arg("rhs"),
            nb::arg("vel")
        )
        .def(
            "comp_divergence",
            [](MLNodeLaplacian& lp, nb::list rhs_py, nb::list vel_py)
            {
                auto n = nb::len(rhs_py);
                Vector<MultiFab*> rhs(n);
                Vector<MultiFab*> vel(n);
                for (size_t i = 0; i < n; ++i)
                {
                    rhs[i] = &nb::cast<MultiFab&>(rhs_py[i]);
                    vel[i] = &nb::cast<MultiFab&>(vel_py[i]);
                }
                lp.compDivergence(rhs, vel);
            },
            nb::arg("rhs"),
            nb::arg("vel")
        );

    // --- MLMG solver ---
    nb::class_<MLMG>(m, "MLMG")
        .def(
            "__init__",
            [](MLMG* self, MLLinOp& lp) { new (self) MLMG(lp); },
            nb::arg("linop"),
            nb::keep_alive<1, 2>()
        )
        .def("set_verbose", &MLMG::setVerbose, nb::arg("v"))
        .def("set_max_iter", &MLMG::setMaxIter, nb::arg("n"))
        .def("set_max_fmg_iter", &MLMG::setMaxFmgIter, nb::arg("n"))
        .def("set_bottom_verbose", &MLMG::setBottomVerbose, nb::arg("v"))
        .def("set_bottom_max_iter", &MLMG::setBottomMaxIter, nb::arg("n"))
        .def("set_bottom_tolerance", &MLMG::setBottomTolerance, nb::arg("t"))
        .def(
            "set_bottom_solver",
            [](MLMG& mlmg, const std::string& which)
            {
                BottomSolver bs = BottomSolver::Default;
                if (which == "default")  bs = BottomSolver::Default;
                else if (which == "smoother") bs = BottomSolver::smoother;
                else if (which == "bicgstab") bs = BottomSolver::bicgstab;
                else if (which == "cg")       bs = BottomSolver::cg;
                else if (which == "bicgcg")   bs = BottomSolver::bicgcg;
                else if (which == "cgbicg")   bs = BottomSolver::cgbicg;
                else throw std::invalid_argument(
                    "unknown bottom solver: " + which);
                mlmg.setBottomSolver(bs);
            },
            nb::arg("which")
        )
        .def(
            "solve",
            [](MLMG& mlmg, MultiFab& sol, const MultiFab& rhs, double rtol, double atol)
            { return mlmg.solve({&sol}, {&rhs}, rtol, atol); },
            nb::arg("sol"),
            nb::arg("rhs"),
            nb::arg("rtol"),
            nb::arg("atol")
        )
        .def(
            "solve",
            [](MLMG& mlmg, nb::list sol_py, nb::list rhs_py, double rtol, double atol)
            {
                auto n = nb::len(sol_py);
                Vector<MultiFab*> sol(n);
                Vector<MultiFab const*> rhs(n);
                for (size_t i = 0; i < n; ++i)
                {
                    sol[i] = &nb::cast<MultiFab&>(sol_py[i]);
                    rhs[i] = &nb::cast<MultiFab const&>(rhs_py[i]);
                }
                return mlmg.solve(sol, rhs, rtol, atol);
            },
            nb::arg("sol"),
            nb::arg("rhs"),
            nb::arg("rtol"),
            nb::arg("atol")
        )
        .def("get_init_residual", &MLMG::getInitResidual)
        .def("get_final_residual", &MLMG::getFinalResidual)
        .def("get_num_iters", &MLMG::getNumIters)
        .def(
            "get_grad_solution",
            [](MLMG& mlmg, MultiFab& gx, MultiFab& gy, MultiFab& gz)
            {
                Array<MultiFab*, AMREX_SPACEDIM> grad = {AMREX_D_DECL(&gx, &gy, &gz)};
                mlmg.getGradSolution({grad});
            },
            nb::arg("gx"),
            nb::arg("gy"),
            nb::arg("gz")
        )
        .def(
            "get_fluxes",
            [](MLMG& mlmg, MultiFab& fluxes)
            {
                mlmg.getFluxes({&fluxes});
            },
            nb::arg("fluxes")
        )
        .def(
            "get_fluxes",
            [](MLMG& mlmg, nb::list fluxes_py)
            {
                auto n = nb::len(fluxes_py);
                Vector<MultiFab*> fluxes(n);
                for (size_t i = 0; i < n; ++i)
                {
                    fluxes[i] = &nb::cast<MultiFab&>(fluxes_py[i]);
                }
                mlmg.getFluxes(fluxes);
            },
            nb::arg("fluxes")
        )
        .def(
            "get_fluxes",
            [](MLMG& mlmg, MultiFab& fx, MultiFab& fy, MultiFab& fz)
            {
                Array<MultiFab*, AMREX_SPACEDIM> fluxes = {AMREX_D_DECL(&fx, &fy, &fz)};
                mlmg.getFluxes({fluxes});
            },
            nb::arg("fx"),
            nb::arg("fy"),
            nb::arg("fz")
        );
}
