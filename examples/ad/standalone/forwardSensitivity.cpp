// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Minimal working example: forward-mode design sensitivities.
//
// Solves 1D steady diffusion with a uniform source on a uniform mesh,
//
//     -d/dx ( nu dphi/dx ) = q,      phi(0) = phiLeft, phi(L) = phiRight
//
// discretised cell-centred finite volume, and reports the exact derivatives of
// two functionals with respect to three design variables:
//
//     alpha = { nu, q, phiLeft }
//     J1    = volume-averaged phi
//     J2    = diffusive flux through the right boundary
//
// The point of the example is that no AD-specific syntax appears in the solver
// or in the functionals. The design variables are declared once; everything
// downstream is ordinary arithmetic on a templated value type, exactly as the
// NeoN operators will be once Phase 1 templating is complete.
//
// Build (no Kokkos, no CMake required):
//   g++ -std=c++20 -I../../../include forwardSensitivity.cpp -o forwardSensitivity

#include "NeoN/ad/designVariables.hpp"
#include "NeoN/core/primitives/dual.hpp"

#include <cstdio>
#include <vector>

namespace
{

constexpr int nAlpha = 3;
using scalar = double;
using Dual = NeoN::Dual<scalar, nAlpha>;

/**
 * @brief Assemble and solve the 1D diffusion system with a Thomas solve.
 *
 * Templated on the value type: instantiated once with Dual to obtain
 * sensitivities and once with plain scalar to produce the finite-difference
 * reference. The mesh spacing dx is deliberately a passive scalar - geometry is
 * frozen, per the agreed scope.
 */
template<typename T>
std::vector<T> solveDiffusion(int nCells, scalar length, T nu, T q, T phiLeft, T phiRight)
{
    const scalar dx = length / static_cast<scalar>(nCells);

    // Tridiagonal system a_i phi_{i-1} + b_i phi_i + c_i phi_{i+1} = r_i
    std::vector<T> a(nCells, T(0)), b(nCells, T(0)), c(nCells, T(0)), r(nCells, T(0));

    const T faceCoeff = nu / T(dx);          // nu * A / d, with A = 1
    const T halfCoeff = nu / T(scalar(0.5) * dx); // boundary face, half distance

    for (int i = 0; i < nCells; ++i)
    {
        r[i] = q * T(dx);

        if (i > 0)
        {
            a[i] = -faceCoeff;
            b[i] += faceCoeff;
        }
        else
        {
            b[i] += halfCoeff;
            r[i] += halfCoeff * phiLeft;
        }

        if (i < nCells - 1)
        {
            c[i] = -faceCoeff;
            b[i] += faceCoeff;
        }
        else
        {
            b[i] += halfCoeff;
            r[i] += halfCoeff * phiRight;
        }
    }

    // Thomas algorithm. Note this is a *direct* solve, so differentiating
    // through it is exact and cheap. For the iterative Ginkgo/PETSc path the
    // implicit function theorem replaces this: one extra linear solve with the
    // same matrix rather than differentiation through the iterations.
    std::vector<T> cp(nCells), rp(nCells);
    cp[0] = c[0] / b[0];
    rp[0] = r[0] / b[0];
    for (int i = 1; i < nCells; ++i)
    {
        const T m = T(1) / (b[i] - a[i] * cp[i - 1]);
        cp[i] = c[i] * m;
        rp[i] = (r[i] - a[i] * rp[i - 1]) * m;
    }

    std::vector<T> phi(nCells);
    phi[nCells - 1] = rp[nCells - 1];
    for (int i = nCells - 2; i >= 0; --i)
    {
        phi[i] = rp[i] - cp[i] * phi[i + 1];
    }
    return phi;
}

/** @brief J1: volume-averaged phi. */
template<typename T>
T meanPhi(const std::vector<T>& phi)
{
    T sum(0);
    for (const auto& p : phi) sum += p;
    return sum / T(static_cast<scalar>(phi.size()));
}

/** @brief J2: diffusive flux through the right boundary face. */
template<typename T>
T outletFlux(const std::vector<T>& phi, scalar length, T nu, T phiRight)
{
    const scalar dx = length / static_cast<scalar>(phi.size());
    return nu * (phiRight - phi.back()) / T(scalar(0.5) * dx);
}

} // namespace

int main()
{
    constexpr int nCells = 64;
    constexpr scalar length = 1.0;
    constexpr scalar phiRightValue = 0.0;

    // --- declare design variables ------------------------------------------
    NeoN::ad::DesignVariables<scalar, nAlpha> dv;

    auto nu = dv.declare("nu", 1.0e-2);
    auto q = dv.declare("sourceStrength", 5.0e-1);
    auto phiLeft = dv.declare("inletValue", 1.0);

    // --- solve and evaluate: no AD-specific syntax below this line ----------
    const auto phi = solveDiffusion<Dual>(nCells, length, nu, q, phiLeft, Dual(phiRightValue));

    const Dual J1 = meanPhi(phi);
    const Dual J2 = outletFlux(phi, length, nu, Dual(phiRightValue));

    // --- finite-difference reference ---------------------------------------
    const scalar base[nAlpha] = {nu.value(), q.value(), phiLeft.value()};

    auto evaluate = [&](const scalar a[nAlpha], int which) -> scalar
    {
        const auto p =
            solveDiffusion<scalar>(nCells, length, a[0], a[1], a[2], phiRightValue);
        return (which == 0) ? meanPhi(p) : outletFlux(p, length, a[0], phiRightValue);
    };

    std::printf("n_alpha = %d, n_J = 2\n\n", dv.size());
    std::printf("J1 (mean phi)     = % .12e\n", J1.value());
    std::printf("J2 (outlet flux)  = % .12e\n\n", J2.value());

    std::printf("%-16s %-8s %18s %18s %12s\n", "variable", "J", "forward AD", "central FD", "rel.err");
    std::printf("%s\n", std::string(78, '-').c_str());

    bool allOk = true;
    for (int which = 0; which < 2; ++which)
    {
        const Dual& J = (which == 0) ? J1 : J2;
        for (int i = 0; i < dv.size(); ++i)
        {
            const scalar h = 1.0e-6 * dv.scale(i);
            scalar plus[nAlpha], minus[nAlpha];
            for (int k = 0; k < nAlpha; ++k) plus[k] = minus[k] = base[k];
            plus[i] += h;
            minus[i] -= h;

            const scalar fd = (evaluate(plus, which) - evaluate(minus, which)) / (2.0 * h);
            const scalar ad = dv.gradient(J, i);
            const scalar den = (std::abs(fd) > 1e-30) ? std::abs(fd) : 1.0;
            const scalar err = std::abs(ad - fd) / den;
            if (err > 1e-6) allOk = false;

            std::printf(
                "%-16s %-8s % 18.10e % 18.10e %12.2e\n",
                dv.name(i).c_str(),
                (which == 0) ? "J1" : "J2",
                ad,
                fd,
                err
            );
        }
    }

    // --- scaled gradient as handed to an optimiser --------------------------
    std::printf("\nscaled gradient of J1 (dimensionless): ");
    for (const auto& g : dv.scaledGradient(J1)) std::printf("% .6e ", g);
    std::printf("\n");

    // --- passivity check ----------------------------------------------------
    const auto dead = dv.passiveVariables(J2);
    if (!dead.empty())
    {
        std::printf("\nWARNING: design variables with zero sensitivity to J2:");
        for (const auto& n : dead) std::printf(" %s", n.c_str());
        std::printf("\n  (no path to the functional - check the case setup)\n");
    }

    std::printf("\n%s\n", allOk ? "PASS: AD matches FD for all entries." : "FAIL");
    return allOk ? 0 : 1;
}
