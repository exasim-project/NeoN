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

#include <iostream>

namespace NeoN::Logging
{

void SupportsLoggingMixin::setLogger(const std::shared_ptr<BaseLogger> logger) { logger_ = logger; }

std::shared_ptr<const BaseLogger> SupportsLoggingMixin::getLogger() const { return logger_; }

namespace
{
// Rank-based logging policy, honoured in BOTH the spdlog and the (default)
// no-spdlog builds. Set once by setNeonDefaultPattern -> never on the hot path.
// On non-root MPI ranks the minimum level is raised to Error, so info/warn/debug
// are muted there, and a "[rank N] " prefix tags the error output that remains.
Level minLevel = Level::Info;
std::string rankPrefix; // empty on rank 0 / serial
}

void setNeonDefaultPattern([[maybe_unused]] mpi::Environment& environment)
{
    const bool nonRoot = environment.isInitialized() && environment.rank() != 0;
    minLevel = nonRoot ? Level::Error : Level::Info;
    rankPrefix = nonRoot ? fmt::format("[rank {}] ", environment.rank()) : std::string {};

#if NF_WITH_SPDLOG
    // logger->set_pattern("%-120v[%^%l%$][%o]");
    auto logger = spdlog::stdout_color_mt("NeoN");
    if (nonRoot)
    {
        // only errors are emitted on non-root ranks; tag them with the rank so
        // distributed error output stays attributable
        logger->set_pattern(fmt::format("[rank {}] %v", environment.rank()));
        logger->set_level(spdlog::level::err);
    }
    else
    {
        // rank 0 (or serial) keeps clean, OpenFOAM-master-style output
        logger->set_pattern("%v");
        logger->set_level(spdlog::level::info);
    }
    logger->info("Initializing NeoN");
#else
    if (shouldLog(Level::Info))
        std::cout << rankPrefix << "Initializing NeoN"
                  << "\n";
#endif
}

bool shouldLog([[maybe_unused]] Level level, [[maybe_unused]] std::string logName)
{
#if NF_WITH_SPDLOG
    auto logger = spdlog::get(logName);
    return logger && logger->should_log(spdlog::level::level_enum(level));
#else
    return level >= minLevel;
#endif
}

void logImpl(std::string sv, [[maybe_unused]] Level level, [[maybe_unused]] std::string logName)
{
#if NF_WITH_SPDLOG
    spdlog::get(logName)->log(spdlog::level::level_enum(level), sv);
#else
    std::cout << rankPrefix << sv << "\n";
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
#if defined(NF_WITH_MPI_SUPPORT) && defined(NF_DEBUG_MESSAGING)
    cpptrace::generate_trace().print();
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    std::terminate();
}

}
