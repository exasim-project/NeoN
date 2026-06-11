// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <source_location>
#include <memory>
#include <string>
#include <string_view>

#include <fmt/core.h>
#include <fmt/chrono.h>

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

/** @brief A class to represent a LogEvent
 *
 */
class LogEvent
{

public:

    LogEvent(std::source_location locationIn, Level levelIn, std::string_view messageIn)
        : location(locationIn), level(levelIn), message(messageIn)
    {
        creationTS = std::chrono::steady_clock::now();
    }

    std::source_location location; // where log message was created

    Level level; // the level the event represents e.g. Trace, Debug, ...

    std::string message; // log message

    std::chrono::time_point<std::chrono::steady_clock> creationTS; // store time of constructor call

    /** @brief convert event to a json string */
    std::string toJson(std::string_view delim)
    {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - creationTS
        );

        return fmt::format(
            fmt::runtime(
                "{{\n\"message\": \"{}\",\n\"sourceLocation\": \"{}:{}\",\n\"timeStarted\": "
                "\"{}\",\n\"duration\": \"{:m}ms\"\n}}{}"
            ),
            message,
            location.file_name(),
            location.line(),
            creationTS.time_since_epoch(),
            duration,
            delim
        );
    }
};

/** Configure the default "NeoN" logger and the rank-based mute policy. Reads the
 * MPI rank internally (when MPI support is enabled and MPI is initialized), so it
 * works with or without MPI and needs no argument. Call after the host has
 * initialized MPI so the rank is known.
 */
void setNeonDefaultPattern();

void logImpl(
    std::string sv, [[maybe_unused]] Level level, [[maybe_unused]] std::string logName = "NeoN"
);

[[noreturn]] void terminate();

namespace detail
{

/** @brief Returns whether a message at the given level would actually be emitted
 *        by the named logger.
 *
 * The convenience helpers below call this BEFORE building the message so that on
 * muted ranks (non-root MPI ranks log at error level only) no string formatting
 * happens at all — keeping the hot path free of cost in distributed runs. It is
 * also null-safe: if the logger is not registered yet it returns false instead
 * of dereferencing a null pointer.
 */
bool shouldLog(Level level, std::string logName = "NeoN");

/* @brief Log a fatal message (rank-tagged) and terminate the program.
 *
 * In MPI runs this aborts the communicator so peers do not deadlock on a
 * collective. Used by NF_ERROR_EXIT / NF_ASSERT (core/error.hpp) and error().
 */
[[noreturn]] void logFatal(const std::string& message);

}

/** @brief convenience function to call spdlogs info with std::format */
template<typename... Args>
void info(std::string formatString, Args... args)
{
    if (!detail::shouldLog(Level::Info)) return;
    logImpl(fmt::format(fmt::runtime(formatString), args...), Level::Info);
}

/** @brief convenience function to call spdlogs warn with std::format */
template<typename... Args>
void warn(std::string formatString, Args... args)
{
    if (!detail::shouldLog(Level::Warning)) return;
    logImpl(fmt::format(fmt::runtime(formatString), args...), Level::Warning);
}

/** @brief convenience function to call spdlogs debug with std::format */
template<typename... Args>
void debug(std::string formatString, Args... args)
{
    if (!detail::shouldLog(Level::Debug)) return;
    logImpl(fmt::format(fmt::runtime(formatString), args...), Level::Debug);
}

namespace detail
{
/** Wraps a format string with the call-site source location. The default
 * source_location argument is evaluated where the implicit conversion from the
 * format string happens — i.e. at the error() call site — so error() captures the
 * caller's file/line without an explicit argument. */
struct FatalFormat
{
    std::string_view fmt;
    std::source_location loc;
    FatalFormat(const char* f, std::source_location l = std::source_location::current())
        : fmt(f), loc(l)
    {}
    FatalFormat(std::string_view f, std::source_location l = std::source_location::current())
        : fmt(f), loc(l)
    {}
    FatalFormat(const std::string& f, std::source_location l = std::source_location::current())
        : fmt(f), loc(l)
    {}
};
}

/** @brief Log a fatal formatted message (with the caller's file:line) and abort. */
template<typename... Args>
[[noreturn]] void error(detail::FatalFormat fmtLoc, Args... args)
{
    auto message = fmt::format(fmt::runtime(fmtLoc.fmt), args...);
    detail::logFatal(fmt::format("{}:{} {}", fmtLoc.loc.file_name(), fmtLoc.loc.line(), message));
}

/** @class A base class to build additional loggers
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

/** @brief convenience function to call log on logger with std::format */
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


/** @class A class for fine-grained logging
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

/** @brief a Logging Mixin class that allows to attach logger to certain classes*/
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
