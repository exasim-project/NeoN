// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

// Deliberately free of both Kokkos and nanobind headers: the implementations live
// in a NON-RDC object library (Kokkos' desul atomics refuse to compile under
// AMReX's -rdc=true) while the nanobind bindings live in the RDC module. This
// header is the only thing the two sides share.

namespace amrex
{
class Geometry;
class MultiFab;
}

namespace blockamr::bench
{

// ---------------------------------------------------------------------------
// Kokkos lifetime, driven from blockamr.initialize()/finalize() so the ordering
// against amrex::Initialize/Finalize is enforced in one place.
// ---------------------------------------------------------------------------
void kokkosInitialize();
void kokkosFinalize();
bool kokkosInitialized();
bool kokkosFinalized();

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
// The native geometric-multigrid V-cycle of solvers/gmg_precond.hpp, run with its
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
    // const fields (the gmg_apply.hpp factory) build one.
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

    // Carry the whole hierarchy in fp32, as production's gmg_precision does. The
    // caller's fields and the residual gate stay fp64; what halves is the traffic of
    // the smoother, which is what the V-cycle is bound by once the launch cost is
    // gone. Unlike the switches above this DOES change arithmetic, so the residual
    // moves in the last few digits. kokkos_opt only -- the baselines stay fp64.
    bool fp32 = false;
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
};

std::vector<std::string> benchGmgBackends();

// batches x iters V-cycles, fenced per batch, each batch restarted from sol = 0.
GmgResult benchGmgVcycle(const std::string& backend, const GmgArgs& args, int iters, int batches);

} // namespace blockamr::bench
