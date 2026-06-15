// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/logging.hpp"

#if NF_WITH_SPDLOG

// See https://github.com/fmtlib/fmt/issues/4610
// NOTE do nothing, TODO only use for AMD clang
auto noAssert = []() {};
#define FMT_ASSERT(condition, message) noAssert();

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/async.h"
#endif

#include <cstdlib>
#include <iostream>

#include "cpptrace/cpptrace.hpp"

namespace NeoN::Logging
{

void SupportsLoggingMixin::setLogger(const std::shared_ptr<BaseLogger> logger) { logger_ = logger; }

std::shared_ptr<const BaseLogger> SupportsLoggingMixin::getLogger() const { return logger_; }

namespace
{
// Rank-based logging policy, honoured in BOTH the spdlog and the (default)
// no-spdlog builds. On non-root MPI ranks the minimum level is raised to Error,
// so info/warn/debug are muted there, and a "[rank N] " prefix tags the error
// output that remains. Resolved once in setNeonDefaultPattern -> shouldLog() is a
// plain comparison with no per-call MPI query (hot-path free, serial included).
// NOTE: for rank-aware muting, MPI must be initialized before NeoN::initialize
// runs (as in neoIcoFoam/neoPisoFoam, which call setRootCase.H first).
Level MIN_LEVEL = Level::Info;
std::string RANK_PREFIX; // empty on rank 0 / serial

void applyRankPolicy(std::size_t rank)
{
    const bool nonRoot = rank != 0;
    MIN_LEVEL = nonRoot ? Level::Error : Level::Info;
    RANK_PREFIX = nonRoot ? fmt::format("[rank {}] ", rank) : std::string {};
}
}

void setNeonDefaultPattern()
{
    // Resolve the rank policy now. In serial (or before MPI is up) the permissive
    // default applies, which is correct for a single-rank run. Done once here so
    // shouldLog() stays a plain comparison with no per-call MPI query.
#ifdef NF_WITH_MPI_SUPPORT
    mpi::Environment environment;
    if (environment.isInitialized()) applyRankPolicy(environment.rank());
#endif

#if NF_WITH_SPDLOG
    auto logger = spdlog::stdout_color_mt("NeoN");
    // Derive pattern/level from the rank policy: RANK_PREFIX is "[rank N] " on
    // non-root ranks (empty otherwise); MIN_LEVEL is Error there so only errors
    // are emitted and stay rank-attributable.
    logger->set_pattern(RANK_PREFIX + "%v");
    logger->set_level(MIN_LEVEL == Level::Error ? spdlog::level::err : spdlog::level::info);
    logger->info("Initializing NeoN");
#else
    if (detail::shouldLog(Level::Info))
        std::cout << RANK_PREFIX << "Initializing NeoN"
                  << "\n";
#endif

    // Report resolved GPU-aware MPI status once at startup (D-07, COMM-05).
    // Must be inside the isInitialized() guard: reading gpuAwareMpi() before MPI is up
    // is undefined behaviour (Environment ctor calls MPI_Comm_rank on an uninit comm).
#ifdef NF_WITH_MPI_SUPPORT
    if (environment.isInitialized())
    {
        const bool gpuAware = environment.gpuAwareMpi();
        const char* gpuAwareMsg = gpuAware
                                    ? "GPU-aware MPI: enabled (device pointers passed to MPI)"
                                    : "GPU-aware MPI: disabled (host staging)";
#if NF_WITH_SPDLOG
        logger->info(gpuAwareMsg);
#else
        if (detail::shouldLog(Level::Info)) std::cout << RANK_PREFIX << gpuAwareMsg << "\n";
#endif
    }
#endif
}

namespace detail
{

bool shouldLog([[maybe_unused]] Level level, [[maybe_unused]] std::string logName)
{
#if NF_WITH_SPDLOG
    auto logger = spdlog::get(logName);
    return logger && logger->should_log(spdlog::level::level_enum(level));
#else
    return level >= MIN_LEVEL;
#endif
}

}

void logImpl(std::string sv, [[maybe_unused]] Level level, [[maybe_unused]] std::string logName)
{
#if NF_WITH_SPDLOG
    spdlog::get(logName)->log(spdlog::level::level_enum(level), sv);
#else
    std::cout << RANK_PREFIX << sv << "\n";
#endif
}

Logger::Logger(std::string name, Level level, Target target)
    : BaseLogger(target), name_(name), level_(level)
{
#if NF_WITH_SPDLOG
    auto logger = target == Target::Console
                    ? spdlog::stdout_color_mt(name_)
                    : spdlog::basic_logger_mt(name_, std::format("{}.json", name_));
    logger->set_pattern("%v");

    if (target == Target::File)
    {
        logImpl("[\n", Level::Info, name_);
    }
#endif
}


void Logger::log(std::string sv) const { logImpl(sv, level_, name_); }


Logger::~Logger()
{
    auto finalizeEvent =
        LogEvent(std::source_location::current(), Logging::Level::Info, "finalizing logger");

    log(finalizeEvent.toJson(""));

    if (getTarget() == Target::File)
    {
        logImpl("]", Level::Info, name_);
    }
}

void terminate()
{
    // Always emit a stack trace (cpptrace is an unconditional dependency).
    cpptrace::generate_trace().print();
#ifdef NF_WITH_MPI_SUPPORT
    // In MPI runs a single failing rank must tear the whole job down, otherwise
    // peers block forever on a collective. Abort the communicator in ALL MPI
    // builds (previously only under debug messaging, and even then commented out).
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    std::abort();
}

namespace detail
{

void logFatal(const std::string& message)
{
    // Fatal path: print directly (rank-tagged) without going through spdlog so it
    // works even before the logger is set up, then abort. Errors are rare, so the
    // direct std::cerr write is fine.
    std::cerr << RANK_PREFIX << message << std::endl;
    terminate();
}

}

}
