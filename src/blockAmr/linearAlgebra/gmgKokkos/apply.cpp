// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The production instantiations of Vcycle: the KokkosGmgApply implementation and apply.hpp's
// factory. The Vcycle template itself lives in vcycle.hpp, shared with the bench.

#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/apply.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/vcycle.hpp"

namespace blockamr
{

namespace
{

// The optimised V-cycle behind apply.hpp's Ginkgo-free handle, fixed to KokkosOptGmgBackend: the
// bench covers the baselines, and a preconditioner has no reason to run an unoptimised launcher.
template<class T, class TC = T>
class KokkosGmgApplyImpl final : public KokkosGmgApply
{
public:

    KokkosGmgApplyImpl(
        const amrex::Geometry& geom, const CallerCoeffs& c, const KokkosGmgOpts& opts
    )
        : v_(geom, c, opts), nCycles_(opts.cycles)
    {}

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
    const ConstCellFieldLevel& alpha,
    const ConstFaceFieldLevel& upper,
    const ConstFaceFieldLevel& lower,
    const KokkosGmgOpts& opts
)
{
    // No bc/geometry check here: la::parseBc already refuses a mismatch and is the only path in.
    // The setup arguments are named once, so each (field, coefficient) pair stays one line; the
    // tag arguments carry the two types, as buildGmgHierarchy's makeGmg does for the shipped one.
    const CallerCoeffs coeffs {
        &*alpha, &upper[0], &lower[0], &upper[1], &lower[1], &upper[2], &lower[2]
    };
    auto make = [&](auto field, auto coeff) -> std::unique_ptr<KokkosGmgApply>
    {
        return std::make_unique<KokkosGmgApplyImpl<decltype(field), decltype(coeff)>>(
            geom, coeffs, opts
        );
    };

    const Precision coeff = parseCoeffPrecision(opts.coeffPrecision, opts.precision);
    switch (precPair(parsePrecision(opts.precision), coeff))
    {
    case PrecPair::f64c32:
        return make(double {}, float {});
    case PrecPair::f64c16:
        return make(double {}, la::Bf16 {});
    case PrecPair::f32c32:
        return make(float {}, float {});
    case PrecPair::f32c16:
        return make(float {}, la::Bf16 {});
    case PrecPair::f16c16:
        return make(la::Bf16 {}, la::Bf16 {});
    case PrecPair::f64c64:
        break;
    }
    return make(double {}, double {});
}

} // namespace blockamr
