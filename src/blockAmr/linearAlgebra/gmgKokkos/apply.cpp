// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The production instantiations of Vcycle: the KokkosGmgApply implementation and
// the factory declared in apply.hpp. Split out of what used to be one TU,
// bench/gmg_vcycle.cpp, together with bench/gmgVcycleBench.cpp (the benchmark
// half) and vcycle.hpp (the Vcycle template itself, shared by both).

#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/apply.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/vcycle.hpp"

namespace blockamr
{

namespace
{

// The optimised V-cycle behind the Ginkgo-free handle of apply.hpp. Fixed to
// KokkosOptGmgBackend: a caller wanting the baselines has the bench for that, and a
// preconditioner has no reason to run a deliberately unoptimised launcher.
template<class T, class TC = T>
class KokkosGmgApplyImpl final : public KokkosGmgApply
{
public:

    KokkosGmgApplyImpl(const GmgArgs& args, int nCycles) : v_(args), nCycles_(nCycles) {}

    void apply(const double* r, double* z) override { v_.applyFlat(r, z, nCycles_); }

    void apply(const float* r, float* z) override { v_.applyFlat(r, z, nCycles_); }

    [[nodiscard]] int nlevels() const override { return v_.nlevels(); }

private:

    Vcycle<KokkosOptGmgBackend, T, TC> v_;
    int nCycles_;
};

} // namespace

std::unique_ptr<KokkosGmgApply> makeKokkosGmgApply(
    const amrex::Geometry& geom,
    const amrex::MultiFab& alpha,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const KokkosGmgOpts& opts
)
{
    // No bc/geometry consistency check here: la::parseBc already refuses a
    // non-periodic direction marked periodic and a periodic one marked otherwise, and
    // it is the only path that reaches this factory. Repeating it would be a branch no
    // test could reach.
    GmgArgs args;
    args.geom = &geom;
    args.rhs = nullptr; // the hierarchy is built from the coefficients alone
    args.alpha = &alpha;
    args.ux = &ux;
    args.lx = &lx;
    args.uy = &uy;
    args.ly = &ly;
    args.uz = &uz;
    args.lz = &lz;
    args.preSweeps = opts.preSweeps;
    args.postSweeps = opts.postSweeps;
    args.coarsestSweeps = opts.coarsestSweeps;
    args.maxLevels = opts.maxLevels;
    args.minBottom = opts.minBottom;
    args.omega = opts.omega;
    args.agglomerate = opts.agglomerate;
    args.aggGridSize = opts.aggGridSize;
    args.precision = opts.precision;
    args.coeffPrecision = opts.coeffPrecision;
    args.shareCoeffs = opts.shareCoeffs;
    args.bc = opts.bc;
    args.aggLevel0Size = opts.aggLevel0Size;

    const Precision coeff = parseCoeffPrecision(opts.coeffPrecision, opts.precision);
    switch (precPair(parsePrecision(opts.precision), coeff))
    {
    case PrecPair::f64c32:
        return std::make_unique<KokkosGmgApplyImpl<double, float>>(args, opts.cycles);
    case PrecPair::f64c16:
        return std::make_unique<KokkosGmgApplyImpl<double, la::Bf16>>(args, opts.cycles);
    case PrecPair::f32c32:
        return std::make_unique<KokkosGmgApplyImpl<float>>(args, opts.cycles);
    case PrecPair::f32c16:
        return std::make_unique<KokkosGmgApplyImpl<float, la::Bf16>>(args, opts.cycles);
    case PrecPair::f16c16:
        return std::make_unique<KokkosGmgApplyImpl<la::Bf16>>(args, opts.cycles);
    case PrecPair::f64c64:
        break;
    }
    return std::make_unique<KokkosGmgApplyImpl<double>>(args, opts.cycles);
}

} // namespace blockamr
