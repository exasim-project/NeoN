// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <mpi.h>

#include "catch2/reporters/catch_reporter_console.hpp"
#include "catch2/reporters/catch_reporter_streaming_base.hpp"
#include "catch2/internal/catch_istream.hpp"

#include "mpiGlobals.hpp"


/**
 * MPI-aware console reporter.
 *
 * This reporter behaves the same as a console reporter, except:
 * - only one thread at a time will print assertion failures
 * - add the rank to the assertion output
 * - only one thread will print the final result, which is gathered from all processes
 */
class MpiReporter : public Catch::StreamingReporterBase
{
public:

    MpiReporter(Catch::ReporterConfig&& config);

    ~MpiReporter() override;

    static std::string getDescription();

    void testRunStarting(const Catch::TestRunInfo& testRunInfo) override;

    void testRunEnded(const Catch::TestRunStats& stats) override;

    void testCaseStarting(const Catch::TestCaseInfo& testInfo) override;

    void testCaseEnded(const Catch::TestCaseStats& stats) override;

    void testCasePartialStarting(const Catch::TestCaseInfo& info, uint64_t uint64) override;

    void testCasePartialEnded(const Catch::TestCaseStats& stats, uint64_t uint64) override;

    void sectionStarting(const Catch::SectionInfo& sectionInfo) override;

    void sectionEnded(const Catch::SectionStats& stats) override;

    void assertionStarting(const Catch::AssertionInfo& info) override;

    void assertionEnded(const Catch::AssertionStats& stats) override;

private:

    Catch::IStream* r_stream_;
    std::unique_ptr<Catch::ConsoleReporter> reporter_;
};
