// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <map>
#include <string>

namespace blockamr::la::prof
{

// Phase profiling. Env var BLOCKAMR_PROFILE (read once):
//   unset/0 : off — one cached-int check per phase, no syncs, no NVTX.
//   1       : wall-clock timers, each phase fenced by streamSynchronize() on both
//             ends (honest attribution, but the syncs perturb the total), + NVTX.
//   2       : NVTX only, no extra syncs — for nsys timelines of an unperturbed solve.
// Accumulated seconds/counts are exposed via profile_report()/profile_reset().

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
