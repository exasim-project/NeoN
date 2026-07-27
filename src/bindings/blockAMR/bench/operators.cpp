// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The Kokkos-vs-AMReX operator bench: 3 cell kernels x 7 launchers, selected by
// name at runtime through NeoN's RuntimeSelectionFactory -- the same abstraction
// NeoN's own schemes use (see surfaceInterpolation.hpp), so what is measured here
// is what a port would actually pay. Dispatch is host-side only: one virtual
// apply() per operator, with a concrete device lambda inside.
//
// Backends come in two families (see launch.hpp): PER-BOX ones launch once per box
// inside an MFIter loop, FUSED ones cover every box in a single launch. A kernel
// body is written once, against a Fields<> bundle of accessors, and both families
// call it -- so no kernel is duplicated to serve a launcher.
//
// This TU is compiled WITHOUT relocatable device code, like the rest of the module
// (see CMakeLists.txt for why _blockamr itself is also non-RDC).

#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <AMReX_MultiFab.H>

#include "NeoN/core/runtimeSelectionFactory.hpp"

#include "../../../blockAmrSolvers/gmgKokkos/launch.hpp"
#include "cells.hpp"
#include "kokkos_bench.hpp"

namespace blockamr::bench
{

// MFIter's destructor stream-synchronizes by default (AMReX_MFIter.cpp:246), a
// host-blocking round trip at the end of every box loop. Leaving it on would make
// this bench measure the wrong thing: the sync waits on AMReX's stream, and Kokkos
// launches on its OWN stream, so the default charges the amrex backend a ~6 us
// synchronization per apply and the Kokkos backends nothing. Both are fenced once
// per batch in benchOperator instead, which times the same work for both.
inline amrex::MFItInfo noSync() { return amrex::MFItInfo().DisableDeviceSync(); }

// ---------------------------------------------------------------------------
// A kernel's inputs, in one device-copyable bundle. Field order is fixed:
// 0 = out, 1 = in, 2..4 = fx, fy, fz. Bundling them is what lets one kernel body
// serve both the per-box and the fused launch families -- the fused one rebuilds
// the bundle per box on the device from AMReX's Array4 table.
// ---------------------------------------------------------------------------

struct Coeffs
{
    double a, cx, cy, cz, dx, dy, dz;
};

Coeffs coeffs(const OpArgs& args)
{
    return Coeffs {
        args.a,
        1.0 / (args.dx * args.dx),
        1.0 / (args.dy * args.dy),
        1.0 / (args.dz * args.dz),
        args.dx,
        args.dy,
        args.dz
    };
}

template<class Acc, int N>
struct Fields
{
    Acc f[N];

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const Acc& operator[](int n) const { return f[n]; }
};

std::array<amrex::MultiFab*, 5> fieldList(const OpArgs& args)
{
    return {args.out, args.in, args.fx, args.fy, args.fz};
}

// ---------------------------------------------------------------------------
// Per-box backends: an accessor type plus a launcher. Every accessor takes GLOBAL
// (i, j, k), so all of them run the identical kernel body from cells.hpp.
// ---------------------------------------------------------------------------

struct AmrexBackend
{
    static constexpr const char* tag = "amrex";
    static constexpr bool fused = false;
    using Acc = amrex::Array4<double>;
    static Acc acc(amrex::FArrayBox& fab) { return fab.array(); }

    template<class F>
    static void launch(const amrex::Box& bx, int, F const& f)
    {
        launchAmrex(bx, f);
    }
};

struct KokkosMdBackend
{
    static constexpr const char* tag = "kokkos_md";
    static constexpr bool fused = false;
    using Acc = ViewAcc;
    static Acc acc(amrex::FArrayBox& fab) { return viewAcc(fab); }

    template<class F>
    static void launch(const amrex::Box& bx, int, F const& f)
    {
        launchKokkosMd(bx, f);
    }
};

struct KokkosFlatBackend
{
    static constexpr const char* tag = "kokkos_flat";
    static constexpr bool fused = false;
    using Acc = ViewAcc;
    static Acc acc(amrex::FArrayBox& fab) { return viewAcc(fab); }

    template<class F>
    static void launch(const amrex::Box& bx, int, F const& f)
    {
        launchKokkosFlat(bx, f);
    }
};

// Diagnostic: Kokkos launcher with AMReX's OWN accessor. The two backends above
// change launcher AND accessor at once, so a gap against amrex cannot be pinned on
// either; this one isolates the launcher. Array4 folds the box origin into its
// pointer at construction, while ViewAcc subtracts it per access and builds a
// Kokkos View per box per apply.
struct KokkosMdArray4Backend
{
    static constexpr const char* tag = "kokkos_md_a4";
    static constexpr bool fused = false;
    using Acc = amrex::Array4<double>;
    static Acc acc(amrex::FArrayBox& fab) { return fab.array(); }

    template<class F>
    static void launch(const amrex::Box& bx, int, F const& f)
    {
        launchKokkosMd(bx, f);
    }
};

// kokkos_md, but round-robined over as many Kokkos streams as AMReX uses. Only one
// thing separates it from kokkos_md, so it answers exactly one question: is the
// multi-box gap AMReX's cross-stream overlap, or something else?
struct KokkosStreamBackend
{
    static constexpr const char* tag = "kokkos_stream";
    static constexpr bool fused = false;
    using Acc = ViewAcc;
    static Acc acc(amrex::FArrayBox& fab) { return viewAcc(fab); }

    template<class F>
    static void launch(const amrex::Box& bx, int ibox, F const& f)
    {
        launchKokkosMdStream(bx, ibox, f);
    }
};

// ---------------------------------------------------------------------------
// Fused backends: one launch for all boxes, so per-box launch cost cannot exist.
// Both use AMReX's cached device Array4 table (mf.arrays()), because the accessor
// was already shown not to matter (kokkos_md_a4 tracks kokkos_md), and sharing it
// keeps the launcher the only difference.
// ---------------------------------------------------------------------------

struct AmrexFusedBackend
{
    static constexpr const char* tag = "amrex_fused";
    static constexpr bool fused = true;
    using Acc = amrex::Array4<double>;

    template<class F>
    static void launchFused(const amrex::MultiFab& mf, F const& f)
    {
        launchAmrexFused(mf, f);
    }
};

struct KokkosTeamBackend
{
    static constexpr const char* tag = "kokkos_team";
    static constexpr bool fused = true;
    using Acc = amrex::Array4<double>;

    template<class F>
    static void launchFused(const amrex::MultiFab& mf, F const& f)
    {
        launchKokkosTeam(mf, f);
    }
};

// ---------------------------------------------------------------------------
// Kernels: each supplies its traffic model, ghost requirement, how many fields it
// reads, and ONE device body over a Fields<> bundle.
// ---------------------------------------------------------------------------

struct AxpyKernel
{
    static constexpr const char* tag = "axpy";
    static constexpr int nghost = 0;
    static constexpr bool needsFaces = false;
    static constexpr double bytesPerCell = 24.0; // read x, read y, write y
    static constexpr int nfields = 2;

    template<class FL>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
    body(FL const& f, int i, int j, int k, const Coeffs& c)
    {
        axpyCell(f[1], f[0], i, j, k, c.a);
    }
};

struct LaplacianKernel
{
    static constexpr const char* tag = "laplacian";
    static constexpr int nghost = 1;
    static constexpr bool needsFaces = false;
    static constexpr double bytesPerCell = 16.0; // stream in + out, ideal caching
    static constexpr int nfields = 2;

    template<class FL>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
    body(FL const& f, int i, int j, int k, const Coeffs& c)
    {
        laplacianCell(f[1], f[0], i, j, k, c.cx, c.cy, c.cz);
    }
};

struct VanLeerKernel
{
    static constexpr const char* tag = "vanleer";
    static constexpr int nghost = 2; // phi(i-2) .. phi(i+2)
    static constexpr bool needsFaces = true;
    static constexpr double bytesPerCell = 40.0; // phi + 3 face fields + out
    static constexpr int nfields = 5;

    template<class FL>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
    body(FL const& f, int i, int j, int k, const Coeffs& c)
    {
        divVanLeerCell(f[1], f[2], f[3], f[4], f[0], i, j, k, c.dx, c.dy, c.dz);
    }
};

// ---------------------------------------------------------------------------
// The two ways to run a kernel over a MultiFab.
// ---------------------------------------------------------------------------

template<class Kernel, class Backend>
void runPerBox(const OpArgs& args)
{
    constexpr int N = Kernel::nfields;
    using Acc = typename Backend::Acc;
    const Coeffs c = coeffs(args);
    const auto mfs = fieldList(args);

    int ibox = 0;
    for (amrex::MFIter mfi(*args.out, noSync()); mfi.isValid(); ++mfi, ++ibox)
    {
        Fields<Acc, N> f {};
        for (int n = 0; n < N; ++n)
        {
            f.f[n] = Backend::acc((*mfs[n])[mfi]);
        }
        Backend::launch(
            mfi.validbox(), ibox, BENCH_LAMBDA(int i, int j, int k) { Kernel::body(f, i, j, k, c); }
        );
    }
}

template<class Kernel, class Backend>
void runFused(const OpArgs& args)
{
    constexpr int N = Kernel::nfields;
    using Acc = typename Backend::Acc;
    const Coeffs c = coeffs(args);
    const auto mfs = fieldList(args);

    // arrays() is cached per FabArray, so this is a device-pointer copy, not a
    // host-to-device transfer per apply.
    Fields<amrex::MultiArray4<double>, N> ma {};
    for (int n = 0; n < N; ++n)
    {
        ma.f[n] = mfs[n]->arrays();
    }

    Backend::launchFused(
        *args.out,
        BENCH_LAMBDA(int b, int i, int j, int k) {
            Fields<Acc, N> f {};
            for (int n = 0; n < N; ++n)
            {
                f.f[n] = ma[n][b];
            }
            Kernel::body(f, i, j, k, c);
        }
    );
}

// ---------------------------------------------------------------------------
// The runtime-selected operator. Parameters<> because operators are stateless --
// all data arrives through apply().
// ---------------------------------------------------------------------------

class CellOperator : public NeoN::RuntimeSelectionFactory<CellOperator, NeoN::Parameters<>>
{
public:

    static std::string name() { return "CellOperator"; }

    CellOperator() = default;

    virtual void apply(const OpArgs& args) const = 0;
    virtual OpInfo info() const = 0;
};

template<class Kernel, class Backend>
class Op : public CellOperator::Register<Op<Kernel, Backend>>
{
public:

    static std::string name() { return std::string(Kernel::tag) + "/" + Backend::tag; }
    static std::string doc() { return name(); }
    static std::string schema() { return "none"; }

    void apply(const OpArgs& args) const override
    {
        if constexpr (Backend::fused)
        {
            runFused<Kernel, Backend>(args);
        }
        else
        {
            runPerBox<Kernel, Backend>(args);
        }
    }

    OpInfo info() const override
    {
        return OpInfo {Kernel::nghost, Kernel::needsFaces, Kernel::bytesPerCell};
    }
};

// Explicit instantiation is what triggers registration -- the same mechanism
// NeoN uses for its own schemes (linear.hpp:107). benchOperators() is asserted
// against this list in the tests, so a silently missing registration fails.
#define BENCH_INSTANTIATE(Kernel)                                                                  \
    template class Op<Kernel, AmrexBackend>;                                                       \
    template class Op<Kernel, AmrexFusedBackend>;                                                  \
    template class Op<Kernel, KokkosMdBackend>;                                                    \
    template class Op<Kernel, KokkosFlatBackend>;                                                  \
    template class Op<Kernel, KokkosMdArray4Backend>;                                              \
    template class Op<Kernel, KokkosStreamBackend>;                                                \
    template class Op<Kernel, KokkosTeamBackend>

BENCH_INSTANTIATE(AxpyKernel);
BENCH_INSTANTIATE(LaplacianKernel);
BENCH_INSTANTIATE(VanLeerKernel);

#undef BENCH_INSTANTIATE

namespace
{

// Fence both runtimes regardless of backend: cheap, and it keeps the timing
// honest without the caller having to know which one launched.
void fenceAll()
{
    amrex::Gpu::streamSynchronize();
    if (Kokkos::is_initialized())
    {
        Kokkos::fence();
    }
}

long validCells(const amrex::MultiFab& mf, int& nboxes)
{
    long n = 0;
    nboxes = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        n += mfi.validbox().numPts();
        ++nboxes;
    }
    return n;
}

} // namespace

std::vector<std::string> benchOperators()
{
    auto names = CellOperator::entries();
    std::sort(names.begin(), names.end());
    return names;
}

OpInfo benchOperatorInfo(const std::string& name) { return CellOperator::create(name)->info(); }

void applyOperator(const std::string& name, const OpArgs& args)
{
    CellOperator::create(name)->apply(args);
    fenceAll();
}

BenchResult benchOperator(const std::string& name, const OpArgs& args, int iters, int batches)
{
    auto op = CellOperator::create(name);

    for (int w = 0; w < 3; ++w)
    {
        op->apply(args);
    }
    fenceAll();

    std::vector<double> ms;
    std::vector<double> msEnq;
    ms.reserve(static_cast<std::size_t>(batches));
    msEnq.reserve(static_cast<std::size_t>(batches));
    for (int b = 0; b < batches; ++b)
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it)
        {
            op->apply(args);
        }
        // t1 is when the HOST is done issuing; t2 is when the device is done. If a
        // backend synchronizes inside its launch, the two collapse together.
        const auto t1 = std::chrono::steady_clock::now();
        fenceAll();
        const auto t2 = std::chrono::steady_clock::now();
        msEnq.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
        ms.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count() / iters);
    }

    std::sort(ms.begin(), ms.end());
    std::sort(msEnq.begin(), msEnq.end());

    BenchResult r;
    r.ncells = validCells(*args.out, r.nboxes);
    r.msMin = ms.front();
    r.msMedian = ms[ms.size() / 2];
    r.msEnqueue = msEnq.front();
    const double bytes = op->info().bytesPerCell * static_cast<double>(r.ncells);
    r.gbPerSec = bytes / (r.msMin * 1.0e-3) / 1.0e9;
    return r;
}

} // namespace blockamr::bench
