// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <string>
#include <vector>

// Deliberately free of both Kokkos and nanobind headers: it is the only thing the
// blockamr_kokkos object library and the _blockamr bindings share.

namespace amrex
{
class Geometry;
class MultiFab;
}

namespace blockamr
{

// Kokkos lifetime lives in runtime.hpp, so init.cpp need not include this bench header.

std::string kokkosExecutionSpace();

// Sum of i over [0, n) -- proves a Kokkos kernel links and launches inside _blockamr.
double kokkosSelftest(long n);

// Sum of mf's valid cells through an unmanaged Kokkos View -- proves the zero-copy handle
// addresses the same bytes AMReX does.
double kokkosMfSum(amrex::MultiFab& mf);

// Fields an operator reads and writes. Ghosts are the caller's job, filled outside the
// timed region. Face fields may be null for operators that do not use them.
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

    // Host time to ISSUE the applies, before the fence: ~ msMin means host-limited (the
    // launch path itself), << msMin means GPU-limited.
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

// batches x iters applies, fenced per batch. Warms up, so msMin excludes PTX JIT.
BenchResult benchOperator(const std::string& name, const OpArgs& args, int iters, int batches);

// The GMG V-cycle bench: gmgPrecond.hpp's native V-cycle with its AMReX kernels and with
// Kokkos twins -- a whole solver phase, so launch cost in a real hierarchy's shape.

// The operator (diagonal source + face coefficients) and the rhs, all FP64 on one
// BoxArray, plus the cycle shape. Mesh must be triply periodic: no physical-BC fill.
struct GmgArgs
{
    // const: the V-cycle only reads these, copying them into its level fields at setup.
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

    // Coarse-grid agglomeration. Off = production, the fine BoxArray coarsened in place,
    // so the coarsest level launches as many kernels as the finest. On = a fresh
    // aggGridSize-capped decomposition when it has fewer boxes. Cost, not arithmetic.
    bool agglomerate = false;
    int aggGridSize = 32;

    // Target box size for LEVEL 0's own decomposition; 0 leaves it on the caller's boxes.
    // Level 0 holds 7/8 of the cells and a box's halo traffic falls as its side grows (19%
    // overhead at 32^3, 9.4% at 64^3). Ignored unless it yields fewer boxes. kokkos_opt.
    int aggLevel0Size = 0;

    // Hierarchy storage type ("fp64" | "fp32" | "bf16"), production's gmg_precision;
    // fields and the residual gate stay fp64. Changes ARITHMETIC (bf16 measurably
    // weaker), so kokkos_opt only -- the baselines stay fp64, which keeps them baselines.
    std::string precision = "fp64";

    // Storage type of the COEFFICIENTS; empty means "same as precision", never WIDER. A
    // rounded coefficient only perturbs the PRECONDITIONER, where a rounded psi is
    // amplified by ||A||. Measured: report/blockamr-precision-measurements.md.
    std::string coeffPrecision;

    // ONE face coefficient per direction instead of an upper/lower pair: for a SYMMETRIC
    // operator the two fabs hold identical numbers, so this is no approximation. Symmetry
    // is CHECKED bitwise at setup; GmgResult::sharedCoeffs reports it. kokkos_opt only.
    bool shareCoeffs = false;

    // Homogeneous domain BCs per side (xlo, xhi, ylo, yhi, zlo, zhi): 0 periodic, 1
    // Dirichlet, 2 Neumann -- la::BcArray respelled to keep this header AMReX-free.
    // All-zero (the default) is the triply periodic mesh the bench itself uses.
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

    // ||rhs - A sol|| before and after ONE V-cycle from sol = 0 -- the correctness gate.
    double resid0 = 0.0;
    double resid1 = 0.0;

    // Whether the hierarchy really shares one coefficient per direction (asymmetric: no).
    bool sharedCoeffs = false;

    // Whether level 0 actually got its own decomposition (GmgArgs::aggLevel0Size).
    bool aggLevel0 = false;
};

std::vector<std::string> benchGmgBackends();

// batches x iters V-cycles, fenced per batch, each batch restarted from sol = 0.
GmgResult benchGmgVcycle(const std::string& backend, const GmgArgs& args, int iters, int batches);

} // namespace blockamr
