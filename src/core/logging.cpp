// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/logging.hpp"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/async.h"

#include <iostream>

namespace NeoN::Logging
{

void SupportsLoggingMixin::setLogger(const std::shared_ptr<BaseLogger> logger) { logger_ = logger; }

const std::shared_ptr<BaseLogger> SupportsLoggingMixin::getLogger() { return logger_; }

void setNeonDefaultPattern()
{
    auto logger = spdlog::stdout_color_mt("NeoN");
    // logger->set_pattern("%-120v[%^%l%$][%o]");
    logger->set_pattern("%v");
    logger->info("Initializing NeoN");
}

void logImpl(std::string sv, Level level, std::string logName)
{
    spdlog::get(logName)->log(spdlog::level::level_enum(level), sv);
}

Logger::Logger(std::string name, Level level, Target target)
    : BaseLogger(target), name_(name), level_(level)
{
    auto logger = target == Target::Console
                    ? spdlog::stdout_color_mt(name_)
                    : spdlog::basic_logger_mt(name_, std::format("{}.json", name_));
    // logger->set_pattern("%-120v[%n][%^%l%$][%o]");
    logger->set_pattern("%v");

    if (target == Target::File)
    {
        logImpl("[\n", Level::Info, name_);
    }
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

}
