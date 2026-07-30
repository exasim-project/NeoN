// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <map>
#include <string>

namespace blockamr::la::prof
{

// Phase profiling, from BLOCKAMR_PROFILE (read once): unset/0 off, 1 wall-clock timers
// with each phase fenced by streamSynchronize (honest, but the syncs perturb) + NVTX,
// 2 NVTX only. Totals are exposed via profile_report()/profile_reset().

int mode();

struct Acc
{
    double sec = 0.0;
    long count = 0;
};

std::map<std::string, Acc>& table();

// Scoped phase timer; lvl >= 0 appends ".L<lvl>" (multigrid level) to the key.
class Timer
{
public:

    explicit Timer(const char* name, int lvl = -1);

    ~Timer();

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

private:

    std::string key_;
    std::chrono::steady_clock::time_point t0_;
};

} // namespace blockamr::la::prof
