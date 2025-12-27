// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <source_location>
#include <format>
#include <memory>
#include <string>
#include <string_view>

#include "NeoN/core/mpi/environment.hpp"

namespace NeoN::Logging
{


enum Level
{
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical
};

enum Target
{
    Console,
    File
};

/* @brief A class to represent a LogEvent
 *
 */
class LogEvent
{

public:

    LogEvent(std::source_location location, Level level, std::string_view message)
        : level(level), message(message), location(location)
    {
        creationTS = std::chrono::steady_clock::now();
    };

    std::source_location location; // where log message was created

    Level level; // the level the event represents e.g. Trace, Debug, ...

    std::string message; // log message

    std::chrono::time_point<std::chrono::steady_clock> creationTS; // store time of constructor call

    /* @brief convert event to a json string */
    std::string toJson(std::string_view delim)
    {
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - creationTS
        );

        return std::format(
            "{{\n\"message\": \"{}\",\n\"sourceLocation\": \"{}:{}\",\n\"timeStarted\": "
            "\"{}\",\n\"duration\": \"{}\"\n}}{}",
            message,
            location.file_name(),
            location.line(),
            creationTS.time_since_epoch(),
            duration,
            delim
        );
    }
};

void setNeonDefaultPattern(NeoN::mpi::Environment& environment);

void logImpl(std::string sv, Level level, std::string logName = "NeoN");

void terminate();

/*@brief convenience function to call spdlogs info with std::format */
template<typename... Args>
void info(std::string formatString, Args... args)
{
    logImpl(std::vformat(formatString, std::make_format_args(args...)), Level::Info);
}

/*@brief convenience function to call spdlogs warn with std::format */
template<typename... Args>
void warn(std::string formatString, Args... args)
{
    logImpl(std::vformat(formatString, std::make_format_args(args...)), Level::Warning);
}

/*@brief convenience function to call spdlogs debug with std::format */
template<typename... Args>
void debug(std::string formatString, Args... args)
{
    logImpl(std::vformat(formatString, std::make_format_args(args...)), Level::Debug);
}

/*@brief convenience function to call spdlogs debug with std::format */
template<typename... Args>
void error(std::string formatString, Args... args)
{
    logImpl(std::vformat(formatString, std::make_format_args(args...)), Level::Error);
    terminate();
}

/* @class A base class to build additional loggers
 */
class BaseLogger
{

    Target target_;

public:

    BaseLogger() : target_(Target::Console) {};

    BaseLogger(Target target) : target_(target) {};

    virtual ~BaseLogger() = default;

    virtual void log(std::string) const = 0;

    Target getTarget() const { return target_; };
};

/*@brief convenience function to call log on logger with std::format */
inline void log(std::shared_ptr<BaseLogger> logger, LogEvent& event, std::string_view delim = ",")
{
    if (logger != nullptr)
    {
        // console output
        auto formattedMessage =
            logger->getTarget() == Target::Console ? event.message : event.toJson(delim);

        logger->log(formattedMessage);
    }
}


/* @class A class for fine-grained logging
 */
class Logger : public BaseLogger
{

public:

    Logger(std::string name, Level level, Target target);

    ~Logger();

    void log(std::string sv) const override;

private:

    // name of the logger in the spdlog registry
    std::string name_;

    Level level_;
};

/*@brief a Logging Mixin class that allows to attach logger to certain classes*/
class SupportsLoggingMixin
{

private:

    std::shared_ptr<BaseLogger> logger_;

public:

    void setLogger(const std::shared_ptr<BaseLogger> logger);

    std::shared_ptr<const BaseLogger> getLogger() const;
};

template<typename CallClass>
void setLogger(CallClass& cls, std::shared_ptr<BaseLogger> logger)
{
    cls.setLogger(logger);
}
}
