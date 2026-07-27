// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "profiling.hpp"

#include <AMReX_GpuLaunch.H>

#if defined(AMREX_USE_CUDA)
#include <nvtx3/nvToolsExt.h> // header-only NVTX v3 (M0 profiling ranges)
#endif

#include <cstdlib>

namespace blockamr::solvers::prof
{

int mode()
{
    static const int m = []
    {
        const char* v = std::getenv("BLOCKAMR_PROFILE");
        return (v != nullptr && v[0] != '\0') ? std::atoi(v) : 0;
    }();
    return m;
}

std::map<std::string, Acc>& table()
{
    static std::map<std::string, Acc> t;
    return t;
}

Timer::Timer(const char* name, int lvl)
{
    if (mode() == 0)
    {
        return;
    }
    key_ = (lvl >= 0) ? std::string(name) + ".L" + std::to_string(lvl) : name;
#if defined(AMREX_USE_CUDA)
    nvtxRangePushA(key_.c_str());
#endif
    if (mode() == 1)
    {
        amrex::Gpu::streamSynchronize();
        t0_ = std::chrono::steady_clock::now();
    }
}

Timer::~Timer()
{
    if (mode() == 0)
    {
        return;
    }
    if (mode() == 1)
    {
        amrex::Gpu::streamSynchronize();
        const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0_;
        auto& a = table()[key_];
        a.sec += dt.count();
        ++a.count;
    }
#if defined(AMREX_USE_CUDA)
    nvtxRangePop();
#endif
}

} // namespace blockamr::solvers::prof
