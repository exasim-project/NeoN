// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/gmgOpts.hpp"

// Deliberately free of both Kokkos and nanobind headers: with gmgOpts.hpp (equally free of them)
// it is all the blockamr_kokkos object library and the _blockamr bindings share.

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

    // The cycle shape, production's own struct: the bench measures the SAME knobs the
    // preconditioner runs. Reduced precision, share_coeffs and aggLevel0Size are honoured by
    // the kokkos_opt backend alone -- benchGmgVcycle refuses them elsewhere rather than
    // reporting an fp64 timing under another label. `cycles` is unused here: benchGmgVcycle
    // takes its own `iters`, so a batch's cycle count is the bench's, not the caller's.
    KokkosGmgOpts opts;
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

    // Whether level 0 actually got its own decomposition (KokkosGmgOpts::aggLevel0Size).
    bool aggLevel0 = false;
};

std::vector<std::string> benchGmgBackends();

// batches x iters V-cycles, fenced per batch, each batch restarted from sol = 0.
GmgResult benchGmgVcycle(const std::string& backend, const GmgArgs& args, int iters, int batches);

} // namespace blockamr
