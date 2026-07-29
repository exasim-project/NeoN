// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <string>
#include <vector>

// Deliberately free of both Kokkos and nanobind headers: the implementations live in
// the blockamr_kokkos object library and the nanobind bindings in _blockamr, and this
// header is the only thing the two sides share.

namespace amrex
{
class Geometry;
class MultiFab;
}

namespace blockamr
{

// Kokkos lifetime lives in runtime.hpp, so production's init.cpp need not include
// this bench-only header.

std::string kokkosExecutionSpace();

// Sum of i over [0, n) -- proves a Kokkos kernel links and launches inside
// _blockamr, with no AMReX interplay involved.
double kokkosSelftest(long n);

// Sum of mf's valid cells through an unmanaged Kokkos View over the fab pointers --
// proves the zero-copy handle addresses the same bytes AMReX does.
double kokkosMfSum(amrex::MultiFab& mf);

// The operator bench.

// Fields an operator reads and writes. Ghosts are the caller's job, filled outside
// the timed region so the halo exchange is not part of the comparison. Face fields
// may be null for operators that do not use them.
struct OpArgs
{
    amrex::MultiFab* out = nullptr;
    amrex::MultiFab* in = nullptr;
    amrex::MultiFab* fx = nullptr;
    amrex::MultiFab* fy = nullptr;
    amrex::MultiFab* fz = nullptr;
    double a = 1.0;  // axpy scalar
    double dx = 1.0; // cell size, per axis
    double dy = 1.0;
    double dz = 1.0;
};

struct BenchResult
{
    double msMin = 0.0;    // fastest batch, per apply (enqueue + completion)
    double msMedian = 0.0; // median batch, per apply
    double gbPerSec = 0.0; // from the operator's ideal traffic model
    long ncells = 0;       // valid cells, summed over boxes
    int nboxes = 0;        // launches per apply

    // Host time to ISSUE the applies, measured before the fence. Separates the two
    // ways a backend can be slow: ~ msMin means host-limited (the launch path
    // itself), << msMin means GPU-limited. Without it, a per-launch-overhead story
    // can be told about either.
    double msEnqueue = 0.0;
};

// What a registered operator needs from the caller before it can run.
struct OpInfo
{
    int nghost = 0;
    bool needsFaces = false;
    double bytesPerCell = 0.0;
};

// Names of every registered operator, as "<kernel>/<backend>".
std::vector<std::string> benchOperators();

OpInfo benchOperatorInfo(const std::string& name);

// One apply, untimed -- the correctness path. Ghosts must already be filled.
void applyOperator(const std::string& name, const OpArgs& args);

// batches x iters applies, fenced per batch. Warms up first, so the reported minimum
// excludes PTX JIT and first-touch effects.
BenchResult benchOperator(const std::string& name, const OpArgs& args, int iters, int batches);

// The GMG V-cycle bench: the native V-cycle of gmgPrecond.hpp, run with its AMReX
// kernels and with Kokkos twins of the same three. Unlike the operator bench this is
// a whole solver phase -- (sweeps x 2 colours + 2) kernels PER LEVEL, each once per
// box, with a ghost exchange between colours -- so it measures launch cost in the
// shape a real multigrid hierarchy produces, coarse levels included.

// The operator (diagonal source + symmetric face coefficients) and the rhs, all FP64
// and sharing one BoxArray/DistributionMapping, plus the V-cycle shape. Mesh must be
// triply periodic: the bench carries no physical-BC ghost fill.
struct GmgArgs
{
    // const: the V-cycle only reads these (it copies them into its own level fields
    // at setup), which is what lets a caller holding const fields build one.
    const amrex::Geometry* geom = nullptr;
    const amrex::MultiFab* rhs = nullptr;
    const amrex::MultiFab* alpha = nullptr;
    const amrex::MultiFab* ux = nullptr;
    const amrex::MultiFab* lx = nullptr;
    const amrex::MultiFab* uy = nullptr;
    const amrex::MultiFab* ly = nullptr;
    const amrex::MultiFab* uz = nullptr;
    const amrex::MultiFab* lz = nullptr;
    int preSweeps = 2;
    int postSweeps = 2;
    int coarsestSweeps = 8;
    int maxLevels = 0; // 0 = coarsen as far as the grid allows
    int minBottom = 2;
    double omega = 1.0;

    // Coarse-grid agglomeration. Off = production: the fine BoxArray is coarsened in
    // place, so the coarsest level launches as many kernels as the finest for a few
    // hundred cells. On = a coarse level takes a fresh aggGridSize-capped
    // decomposition whenever that has fewer boxes, with inter-level kernels routed
    // through a transfer fab. Red-black smoothing is decomposition-independent, so
    // this changes cost and not arithmetic.
    bool agglomerate = false;
    int aggGridSize = 32;

    // Target box size for LEVEL 0's own decomposition; 0 (the default) leaves level 0
    // on the caller's boxes. Re-deciding the one decomposition the caller can see
    // costs a staging fab and a copy per apply, but level 0 holds 7/8 of the cells and
    // a box's halo traffic falls as its side grows (19% overhead at 32^3, 9.4% at
    // 64^3). Ignored unless it yields strictly fewer boxes. kokkos_opt only.
    int aggLevel0Size = 0;

    // The value type the whole hierarchy is STORED in, as production's
    // gmg_precision: "fp64" (the default), "fp32" or "bf16". The caller's fields and
    // the residual gate stay fp64 whatever this says; what shrinks is smoother
    // traffic, which is what the V-cycle is bound by once launch cost is gone.
    //
    // Unlike the switches above these DO change arithmetic. bf16's cycle is
    // measurably weaker, more so the finer the grid, since the restricted residual
    // carries psi's storage error times ||A|| ~ 6/dx^2 (1.05x weaker at 16^3, 3.2x at
    // 256^3) -- more CG iterations than the saved bytes buy back at any size.
    // Arithmetic still happens in fp32; in bf16 the residual would cancel to exactly
    // zero. kokkos_opt only, for all three: the baselines stay fp64, which is what
    // keeps them baselines.
    std::string precision = "fp64";

    // The value type the COEFFICIENTS (alpha and the face arrays) are stored in;
    // empty means "same as precision". Split from `precision` because a rounded psi is
    // amplified by ||A|| when the cycle restricts b - A psi, while a rounded
    // coefficient only perturbs the PRECONDITIONER's operator -- CG still stops on the
    // fp64 one, so the cost is iterations, not correctness. kokkos_opt only, and it
    // may not be WIDER than precision.
    //
    // Measured: narrow the coefficients only once the FIELDS are narrow. Under fp32
    // fields, bf16 coefficients gain 1.18x at an indistinguishable residual
    // reduction; under fp64 fields the same change costs 11%. Tables:
    // report/blockamr-precision-measurements.md.
    std::string coeffPrecision;

    // Store ONE face coefficient per direction instead of an upper/lower pair: for a
    // SYMMETRIC operator ux(i+1) and lx(i+1) are the same matrix entry, so the two
    // fabs hold identical numbers and a colour sweep streams 9 arrays where 6 suffice.
    // Not an approximation -- the same numbers with half the coefficient traffic.
    //
    // Symmetry is CHECKED bitwise at setup rather than assumed, and an asymmetric
    // operator silently keeps the pair; GmgResult::sharedCoeffs reports which
    // happened, so a timing can never be labelled shared when it was not.
    // kokkos_opt only.
    bool shareCoeffs = false;

    // Homogeneous domain BCs per side (xlo, xhi, ylo, yhi, zlo, zhi): 0 periodic,
    // 1 Dirichlet, 2 Neumann -- la::BcArray's spec, respelled here so this header
    // keeps its no-AMReX-headers contract. All-zero (the default) is the triply
    // periodic mesh the bench itself uses.
    std::array<int, 6> bc {};
};

struct GmgResult
{
    double msMin = 0.0;    // fastest batch, per V-cycle
    double msMedian = 0.0; // median batch, per V-cycle
    double msEnqueue = 0.0;

    int nlevels = 0;
    std::vector<int> boxesPerLevel;
    std::vector<long> cellsPerLevel;

    // ||rhs - A sol|| before and after ONE V-cycle from sol = 0: the gate that says
    // the timed launcher computed the V-cycle rather than something cheaper.
    double resid0 = 0.0;
    double resid1 = 0.0;

    // Whether the hierarchy actually shares one face coefficient per direction:
    // asking for it on an asymmetric operator does not get it.
    bool sharedCoeffs = false;

    // Whether level 0 actually got its own decomposition (GmgArgs::aggLevel0Size).
    bool aggLevel0 = false;
};

std::vector<std::string> benchGmgBackends();

// batches x iters V-cycles, fenced per batch, each batch restarted from sol = 0.
GmgResult benchGmgVcycle(const std::string& backend, const GmgArgs& args, int iters, int batches);

} // namespace blockamr
