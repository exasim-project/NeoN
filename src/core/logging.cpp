// SPDX-FileCopyrightText: 2025 NeoN authors
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

void setNeonDefaultPattern(mpi::Environment& environment)
{
    std::cout << __FILE__ << ":" << __LINE__ << "setPattern\n";
#if NF_WITH_SPDLOG
    // logger->set_pattern("%-120v[%^%l%$][%o]");
    auto logger = spdlog::stdout_color_mt("NeoN");
    logger->set_pattern("%v");
    logger->set_level(spdlog::level::info);
    // mute non rank zero output
    std::cout << __FILE__ << ":" << __LINE__ << "environment.isInitialized()"
              << environment.isInitialized() << "environment.rank()" << environment.rank() << "\n";

    if (environment.isInitialized() && environment.rank() != 0)
    {
        logger->set_level(spdlog::level::err);
    }
    logger->info("Initializing NeoN");
#endif
}

void logImpl(std::string sv, Level level, std::string logName)
{
#if NF_WITH_SPDLOG
    spdlog::get(logName)->log(spdlog::level::level_enum(level), sv);
#else
    std::cout << sv << "\n";
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
#ifdef NF_WITH_MPI_SUPPORT
    cpptrace::generate_trace().print();
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    std::terminate();
}

}
