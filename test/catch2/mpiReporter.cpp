// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#include "mpiReporter.hpp"

#include "catch2/catch_get_random_seed.hpp"
#include "catch2/catch_test_case_info.hpp"
#include "catch2/reporters/catch_reporter_streaming_base.hpp"
#include "catch2/reporters/catch_reporter_registrars.hpp"
#include "catch2/reporters/catch_reporter_console.hpp"
#include "catch2/internal/catch_istream.hpp"
#include "catch2/internal/catch_context.hpp"
#include "catch2/internal/catch_random_number_generator.hpp"

#include "mpiSerialization.hpp"


/**
 * Stream that doesn't do anything (think of /dev/null)
 */
class NullStream : public std::ostream
{
};

/**
 * Make sure that writing to NullStream doesn't to anything
 */
template<class T>
const NullStream& operator<<(NullStream&& os, const T& value)
{
    return os;
}

/**
 * Catch stream that doesn't do anything
 *
 * This is necessary, since catch streams can't be copied, but the MpiReporter needs two streams,
 * one to initialize the base class, one to initialize the wrapped stream.
 * A new DiscardStream is passed to the base class, while the real stream is passed to the wrapped
 * stream.
 */
class DiscardStream : public Catch::IStream
{
public:

    std::ostream& stream() override { return nullStream; }

private:

    NullStream nullStream;
};


MpiReporter::MpiReporter(Catch::ReporterConfig&& config)
    : Catch::StreamingReporterBase(Catch::ReporterConfig(
        config.fullConfig(),
        Catch::Detail::unique_ptr<DiscardStream>(new DiscardStream),
        config.colourMode(),
        config.customOptions()
    ))
{
    auto colorMode = config.colourMode();
    auto istream = CATCH_MOVE(config).takeStream();
    r_stream_ = istream.get();

    reporter_ = std::make_unique<Catch::ConsoleReporter>(Catch::ReporterConfig {
        this->m_config, CATCH_MOVE(istream), colorMode, this->m_customOptions
    });

    this->m_preferences.shouldReportAllAssertions = true;

    // Trigger the lazy printer in the nested reporter
    if (!IS_ROOT)
    {
        // disable std stream temporarily
        r_stream_->stream().setstate(std::ios_base::badbit);

        // need to provide actual name, not empty string
        Catch::SourceLineInfo lineInfo("dummy.cpp", 0);
        Catch::TestCaseInfo testCaseInfo({}, {}, lineInfo);
        Catch::TestCaseStats testCaseStats(testCaseInfo, {}, {}, {}, {});

        Catch::SectionInfo sectionInfo(lineInfo, {}, {});
        Catch::SectionStats sectionStats(Catch::SectionInfo(sectionInfo), {}, {}, true);

        reporter_->testCaseStarting(testCaseInfo);
        reporter_->sectionStarting(sectionInfo);
        reporter_->sectionEnded(sectionStats);
        reporter_->testCaseEnded(testCaseStats);

        // re-enable std stream
        r_stream_->stream().clear();
    }
}

MpiReporter::~MpiReporter() {}

std::string MpiReporter::getDescription() { return "MPI aware console reporter"; }

void MpiReporter::testRunStarting(const Catch::TestRunInfo& testRunInfo)
{
    //  use the same RNG seed on all processes
    auto seed = Catch::getSeed();
    MPI_Bcast(&seed, 1, MPI_INT, 0, COMM);
    Catch::sharedRng().seed(seed);
    if (IS_ROOT)
    {
        reporter_->testRunStarting(testRunInfo);
    }
}

void MpiReporter::testRunEnded(const Catch::TestRunStats& stats)
{
    Catch::TestRunStats globalStats(stats);
    MPI_Reduce(&stats.aborting, &globalStats.aborting, 1, MPI_CXX_BOOL, MPI_LOR, ROOT, COMM);
    MPI_Reduce(
        &stats.totals.assertions.failed,
        &globalStats.totals.assertions.failed,
        1,
        MPI_UINT64_T,
        MPI_MAX,
        ROOT,
        COMM
    );
    MPI_Reduce(
        &stats.totals.assertions.passed,
        &globalStats.totals.assertions.passed,
        1,
        MPI_UINT64_T,
        MPI_MIN,
        ROOT,
        COMM
    );
    MPI_Reduce(
        &stats.totals.testCases.failed,
        &globalStats.totals.testCases.failed,
        1,
        MPI_UINT64_T,
        MPI_MAX,
        ROOT,
        COMM
    );
    MPI_Reduce(
        &stats.totals.testCases.passed,
        &globalStats.totals.testCases.passed,
        1,
        MPI_UINT64_T,
        MPI_MIN,
        ROOT,
        COMM
    );
    // Assume [assertion|testCases].skipped is the same for each process
    // Ignore [assertion|testCases].failedButOk, since I don't know what that means
    if (IS_ROOT)
    {
        reporter_->testRunEnded(globalStats);
    }
}

void MpiReporter::testCaseStarting(const Catch::TestCaseInfo& testInfo)
{
    reporter_->testCaseStarting(testInfo);
}

void MpiReporter::testCaseEnded(const Catch::TestCaseStats& stats)
{
    reporter_->testCaseEnded(stats);
}

void MpiReporter::testCasePartialStarting(const Catch::TestCaseInfo& info, uint64_t uint64)
{
    reporter_->testCasePartialStarting(info, uint64);
}

void MpiReporter::testCasePartialEnded(const Catch::TestCaseStats& stats, uint64_t uint64)
{
    reporter_->testCasePartialEnded(stats, uint64);
}

void MpiReporter::sectionStarting(const Catch::SectionInfo& sectionInfo)
{
    reporter_->sectionStarting(sectionInfo);
}

void MpiReporter::sectionEnded(const Catch::SectionStats& stats) { reporter_->sectionEnded(stats); }

void MpiReporter::assertionStarting(const Catch::AssertionInfo& info)
{
    reporter_->assertionStarting(info);
}

void MpiReporter::assertionEnded(const Catch::AssertionStats& stats)
{
    const bool needPrint = !stats.assertionResult.isOk();

    if (needPrint)
    {
        MPI_Send(&needPrint, 1, MPI_CXX_BOOL, ROOT, SERIALIZATION_TAG, COMM);

        bool allowedToPrint = false;
        MPI_Recv(
            &allowedToPrint, 1, MPI_CXX_BOOL, ROOT, SERIALIZATION_TAG, COMM, MPI_STATUS_IGNORE
        );

        r_stream_->stream() << "Rank [" << RANK << "|" << COMM_SIZE << "]\n" << std::flush;
        reporter_->assertionEnded(stats);
        r_stream_->stream() << std::flush;

        const bool finished = true;
        MPI_Send(&finished, 1, MPI_CXX_BOOL, ROOT, SERIALIZATION_TAG, COMM);
    }
}
