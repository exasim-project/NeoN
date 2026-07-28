// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <string>
#include <vector>

// Deliberately free of both Kokkos and nanobind headers: the implementations live in
// the blockamr_kokkos object library while the nanobind bindings live in _blockamr --
// separate libraries by history, not by an RDC fence (see CMakeLists.txt for why the
// whole module builds non-RDC). This header is the only thing the two sides share.

namespace amrex
{
class Geometry;
class MultiFab;
}

namespace blockamr::bench
{

// Kokkos lifetime (kokkosInitialize/kokkosFinalize/kokkosInitialized/
// kokkosFinalized) now lives in runtime.hpp (blockAmr/kokkos/) --
// production's init.cpp includes that instead of this bench-only contract header.

std::string kokkosExecutionSpace();

// Sum of i over [0, n) on the default execution space -- proves a Kokkos kernel
// links and launches inside _blockamr, with no AMReX interplay involved.
double kokkosSelftest(long n);

// Sum of mf's valid cells, reduced through an unmanaged Kokkos View over the fab
// pointers -- proves the zero-copy handle addresses the same bytes AMReX does.
double kokkosMfSum(amrex::MultiFab& mf);

// ---------------------------------------------------------------------------
// The operator bench
// ---------------------------------------------------------------------------

// Fields an operator reads and writes. Ghost cells are the caller's job (filled
// once, outside the timed region, so the halo exchange is not part of the
// comparison). Face fields may be null for operators that do not use them.
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
    // ways a backend can be slow: msEnqueue ~ msMin means the run is host-limited
    // (the launch path itself), msEnqueue << msMin means it is GPU-limited and any
    // gap is on the device. Without this the two cannot be told apart, and a
    // per-launch-overhead story can be told about either.
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

// batches x iters applies, fenced per batch. Warms up first, so the reported
// minimum excludes PTX JIT and first-touch effects.
BenchResult benchOperator(const std::string& name, const OpArgs& args, int iters, int batches);

// ---------------------------------------------------------------------------
// The GMG V-cycle bench
//
// The native geometric-multigrid V-cycle of gmgPrecond.hpp, run with its
// AMReX kernels and with Kokkos twins of the same three kernels. Unlike the
// operator bench this is a whole solver phase: per V-cycle it launches
// (sweeps x 2 colours + 2) kernels PER LEVEL, each once per box, with a ghost
// exchange between colours -- so it measures launch cost in the shape a real
// multigrid hierarchy produces, coarse levels included.
// ---------------------------------------------------------------------------

// The operator (OpenFOAM-style diagonal source + symmetric face coefficients) and
// the right-hand side, all FP64 and sharing one BoxArray/DistributionMapping, plus
// the V-cycle shape. Mesh must be triply periodic: the bench does not carry the
// production physical-BC ghost fill.
struct GmgArgs
{
    // const: the V-cycle only ever reads these — it copies them into its own level
    // fields at setup — and a const-correct GmgArgs is what lets a caller holding
    // const fields (the apply.hpp factory) build one.
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

    // Coarse-grid agglomeration. Off = production: the fine BoxArray is coarsened
    // in place, so every level has the SAME box count and the coarsest level
    // launches as many kernels as the finest for a few hundred cells. On = a coarse
    // level whose in-place decomposition would have more boxes than a fresh
    // aggGridSize-capped decomposition of its domain uses the fresh one instead,
    // and the inter-level kernels route through a transfer fab on the fine level's
    // layout. Red-black smoothing is decomposition-independent, so this changes
    // cost and not arithmetic -- at equal depth the residual is unchanged.
    bool agglomerate = false;
    int aggGridSize = 32;

    // Target box size for LEVEL 0's own decomposition; 0 (the default) leaves level 0
    // on the caller's boxes, which is what every level did before. Level 0 is the one
    // decomposition the caller can see, so re-deciding it costs a staging fab and a
    // copy at each end of an apply -- but level 0 holds 7/8 of the hierarchy's cells,
    // and a box's halo traffic falls as its side grows (6*32^2 ghosts per 32^3
    // interior is 19% overhead; 64^3 is 9.4%). Ignored unless it yields strictly
    // fewer boxes than the caller's. kokkos_opt only.
    int aggLevel0Size = 0;

    // The value type the whole hierarchy is STORED in, as production's
    // gmg_precision: "fp64" (the default), "fp32" or "bf16". The caller's fields
    // and the residual gate stay fp64 whatever this says; what shrinks is the
    // traffic of the smoother, which is what the V-cycle is bound by once the
    // launch cost is gone -- half at fp32, a quarter at bf16.
    //
    // Unlike the switches above these DO change arithmetic. fp32 moves the
    // residual in the last few digits; bf16 keeps only ~3 decimal digits per
    // stored value, and the resulting cycle is measurably weaker -- more so the
    // finer the grid, since the restricted residual carries psi's storage error
    // multiplied by ||A|| ~ 6/dx^2 (1.05x weaker at 16^3, 3.2x at 256^3), which
    // costs more CG iterations than the saved bytes buy back at any size. Its
    // arithmetic still happens in fp32 (solvers::GmgComputeT, bf16.hpp); in bf16
    // the residual would cancel to exactly zero. kokkos_opt only, for all three:
    // the baselines stay fp64.
    std::string precision = "fp64";

    // The value type the COEFFICIENTS (alpha and the face arrays) are stored in;
    // empty means "same as precision", which is what every level did before this
    // knob existed. The split is the refinement the bf16 measurement pointed at: a
    // rounded psi is amplified by ||A|| when the cycle restricts b - A psi, while a
    // rounded coefficient is only a perturbation of the PRECONDITIONER's operator --
    // CG still stops on the fp64 one, so the cost is iterations, not correctness.
    // With shareCoeffs on, the coefficients are 4 of the 6 arrays a colour sweep
    // streams, so narrowing them alone moves most of the bytes bf16 was after.
    // kokkos_opt only, and it may not be WIDER than precision.
    //
    // Measured: narrow the coefficients only once the FIELDS are narrow. Under fp32
    // fields bf16 coefficients are 1.18x off the cycle at a residual reduction
    // indistinguishable from fp32's; under fp64 fields the same change costs 11%.
    // Tables: report/blockamr-precision-measurements.md in the NeoFOAM repo.
    std::string coeffPrecision;

    // Store ONE face coefficient per direction instead of an upper/lower pair.
    // ux(i+1,j,k) is cell i's east coefficient and lx(i+1,j,k) is cell i+1's west
    // coefficient -- for a SYMMETRIC operator those are the same matrix entry, so
    // the two fabs hold identical numbers and a colour sweep streams 9 arrays where
    // 6 suffice. When on, the level allocates ux/uy/uz only and the kernels read the
    // east coefficient at face i+1 and the west at face i of that one array: not an
    // approximation, the same numbers with half the coefficient traffic.
    //
    // Symmetry is CHECKED (bitwise, at setup) rather than assumed, and an asymmetric
    // operator silently keeps the pair -- GmgResult::sharedCoeffs reports which
    // happened, so a timing can never be labelled as shared when it was not.
    // kokkos_opt only.
    bool shareCoeffs = false;

    // Homogeneous domain boundary conditions per side (xlo, xhi, ylo, yhi, zlo, zhi):
    // 0 periodic, 1 Dirichlet, 2 Neumann -- the same spec (and the same type as)
    // solvers::BcArray, spelled out here so this header keeps its no-AMReX-headers
    // contract. All-zero (the default) is the triply periodic mesh the bench itself
    // uses, where the boundary fill has nothing to do.
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

    // ||rhs - A sol|| before and after ONE V-cycle from sol = 0. The gate that says
    // the timed launcher computed the V-cycle rather than something cheaper.
    double resid0 = 0.0;
    double resid1 = 0.0;

    // Whether the hierarchy actually shares one face coefficient per direction (see
    // GmgArgs::shareCoeffs): asking for it on an asymmetric operator does not get it.
    bool sharedCoeffs = false;

    // Whether level 0 actually got its own decomposition (GmgArgs::aggLevel0Size).
    bool aggLevel0 = false;
};

std::vector<std::string> benchGmgBackends();

// batches x iters V-cycles, fenced per batch, each batch restarted from sol = 0.
GmgResult benchGmgVcycle(const std::string& backend, const GmgArgs& args, int iters, int batches);

} // namespace blockamr::bench
