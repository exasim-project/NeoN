// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoN/core/mpi/operators.hpp"

using namespace NeoN;
using namespace NeoN::mpi;

TEST_CASE("getOp")
{
    REQUIRE(getOp(ReduceOp::Max) == MPI_MAX);
    REQUIRE(getOp(ReduceOp::Min) == MPI_MIN);
    REQUIRE(getOp(ReduceOp::Sum) == MPI_SUM);
    REQUIRE(getOp(ReduceOp::Prod) == MPI_PROD);
    REQUIRE(getOp(ReduceOp::Land) == MPI_LAND);
    REQUIRE(getOp(ReduceOp::Band) == MPI_BAND);
    REQUIRE(getOp(ReduceOp::Lor) == MPI_LOR);
    REQUIRE(getOp(ReduceOp::Bor) == MPI_BOR);
    REQUIRE(getOp(ReduceOp::Maxloc) == MPI_MAXLOC);
    REQUIRE(getOp(ReduceOp::Minloc) == MPI_MINLOC);
}

TEST_CASE("getType")
{
    REQUIRE(getType<char>() == MPI_CHAR);
    REQUIRE(getType<wchar_t>() == MPI_WCHAR);
    REQUIRE(getType<short>() == MPI_SHORT);
    REQUIRE(getType<int>() == MPI_INT);
    REQUIRE(getType<long>() == MPI_LONG);
    REQUIRE(getType<long long>() == MPI_LONG_LONG);
    REQUIRE(getType<unsigned>() == MPI_UNSIGNED);
    REQUIRE(getType<unsigned long>() == MPI_UNSIGNED_LONG);
    REQUIRE(getType<unsigned long long>() == MPI_UNSIGNED_LONG_LONG);
    REQUIRE(getType<float>() == MPI_FLOAT);
    REQUIRE(getType<double>() == MPI_DOUBLE);
    REQUIRE(getType<long double>() == MPI_LONG_DOUBLE);
    REQUIRE(getType<bool>() == MPI_CXX_BOOL);
    REQUIRE(getType<std::complex<float>>() == MPI_CXX_FLOAT_COMPLEX);
    REQUIRE(getType<std::complex<double>>() == MPI_CXX_DOUBLE_COMPLEX);
    REQUIRE(getType<std::complex<long double>>() == MPI_CXX_LONG_DOUBLE_COMPLEX);
}

TEST_CASE("allReduce value")
{
    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    SECTION("MPI_MAX")
    {
        int value = rank;
        int sendValue = value;
        allReduce(sendValue, ReduceOp::Max, MPI_COMM_WORLD);
        REQUIRE(sendValue == (ranks - 1));
    }

    SECTION("MPI_MIN")
    {
        int value = rank;
        int sendValue = value;
        allReduce(sendValue, ReduceOp::Min, MPI_COMM_WORLD);
        REQUIRE(sendValue == 0);
    }

    SECTION("MPI_SUM")
    {
        int value = 2;
        int sendValue = value;
        allReduce(sendValue, ReduceOp::Sum, MPI_COMM_WORLD);
        REQUIRE(sendValue == value * ranks);
    }

    SECTION("MPI_PROD")
    {
        float value = 2.5f;
        float sendValue = value;
        allReduce(sendValue, ReduceOp::Prod, MPI_COMM_WORLD);
        REQUIRE(sendValue == std::pow(value, static_cast<float>(ranks)));
    }

    SECTION("MPI_LAND")
    {
        int value = (rank == 0) ? 1 : 0;
        int sendValue = value;
        allReduce(sendValue, ReduceOp::Land, MPI_COMM_WORLD);
        REQUIRE(sendValue == 0);
    }

    SECTION("MPI_LOR")
    {
        int value = (rank == 0) ? 1 : 0;
        int sendValue = value;
        allReduce(sendValue, ReduceOp::Lor, MPI_COMM_WORLD);
        REQUIRE(sendValue == 1);
    }

    SECTION("MPI_BAND")
    {
        int value = (rank == 0) ? 1 : 0;
        int sendValue = value;
        allReduce(sendValue, ReduceOp::Band, MPI_COMM_WORLD);
        REQUIRE(sendValue == 0);
    }

    SECTION("MPI_BOR")
    {
        int value = (rank == 0) ? 1 : 0;
        int sendValue = value;
        allReduce(sendValue, ReduceOp::Bor, MPI_COMM_WORLD);
        REQUIRE(sendValue == 1);
    }
}

TEST_CASE("allReduce vectors")
{
    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);


    SECTION("MPI_MAX")
    {
        Vec3 values;
        for (size_t ivalue = 0; ivalue < values.size(); ++ivalue)
        {
            values[ivalue] = static_cast<scalar>(rank) + static_cast<scalar>(ivalue) * 10.0;
        }
        Vec3 sendValues = values;
        allReduce(sendValues, ReduceOp::Max, MPI_COMM_WORLD);
        for (size_t ivalue = 0; ivalue < values.size(); ++ivalue)
        {
            scalar expectedMaxValue =
                static_cast<scalar>(ranks - 1) + static_cast<scalar>(ivalue) * 10.0;
            REQUIRE(sendValues[ivalue] == expectedMaxValue);
        }
    }

    SECTION("MPI_MIN")
    {
        Vec3 values;
        for (size_t ivalue = 0; ivalue < values.size(); ++ivalue)
        {
            values[ivalue] = static_cast<scalar>(rank) + static_cast<scalar>(ivalue) * 10.0;
        }
        Vec3 sendValues = values;
        allReduce(sendValues, ReduceOp::Min, MPI_COMM_WORLD);
        for (size_t ivalue = 0; ivalue < values.size(); ++ivalue)
        {
            REQUIRE(sendValues[ivalue] == static_cast<scalar>(ivalue) * 10.0);
        }
    }

    SECTION("MPI_SUM")
    {
        Vec3 values;
        for (size_t ivalue = 0; ivalue < values.size(); ++ivalue)
        {
            values[ivalue] = static_cast<scalar>(ivalue) * 10.0;
        }
        Vec3 sendValues = values;
        allReduce(sendValues, ReduceOp::Sum, MPI_COMM_WORLD);
        for (size_t ivalue = 0; ivalue < values.size(); ++ivalue)
        {
            values[ivalue] = static_cast<scalar>(ranks) * static_cast<scalar>(ivalue) * 10.0;
        }
        for (size_t ivalue = 0; ivalue < sendValues.size(); ++ivalue)
        {
            REQUIRE(sendValues[ivalue] == values[ivalue]);
        }
    }

    SECTION("MPI_PROD")
    {
        Vec3 values;
        for (size_t ivalue = 0; ivalue < values.size(); ++ivalue)
        {
            values[ivalue] = static_cast<scalar>(ivalue) * 10.0;
        }
        Vec3 sendValues = values;
        allReduce(sendValues, ReduceOp::Prod, MPI_COMM_WORLD);
        for (size_t ivalue = 0; ivalue < values.size(); ++ivalue)
        {
            values[ivalue] = std::pow(values[ivalue], static_cast<float>(ranks));
        }
        for (size_t ivalue = 0; ivalue < sendValues.size(); ++ivalue)
        {
            REQUIRE(sendValues[ivalue] == values[ivalue]);
        }
    }
}

TEST_CASE("isend irecv Test")
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    // data packet to be sent from rank 0 to rank 1
    const int size01 = 3;
    int sendValue01[size01] = {42, -1, 145}; // Values known to all ranks so we can check.

    // data packet to be sent from rank 1 to rank 0
    const int size10 = 2;
    int sendValue10[size10] = {-58, 234}; // Values known to all ranks so we can check.
    const int tag = 0;
    MPI_Request requestSend;
    MPI_Request requestReceive;

    if (rank == 0)
    {
        int sendBuffer[size01];
        int recvBuffer[size10];
        std::copy(std::begin(sendValue01), std::end(sendValue01), std::begin(sendBuffer));
        isend(sendBuffer, size01, 1, tag, comm, &requestSend);    // we send this data to rank 1
        irecv(recvBuffer, size10, 1, tag, comm, &requestReceive); // we into this buffer from rank 1

        while (!test(&requestSend))
        {
            // Busy wait
        }

        while (!test(&requestReceive))
        {
            // Busy wait
        }

        // Check the received value
        for (int i = 0; i < size10; i++)
            REQUIRE(recvBuffer[i] == sendValue10[i]);
    }
    else if (rank == 1)
    {

        int sendBuffer[size10];
        int recvBuffer[size01];
        std::copy(std::begin(sendValue10), std::end(sendValue10), std::begin(sendBuffer));
        isend(sendBuffer, size10, 0, tag, comm, &requestSend);    // we send this data to rank 0
        irecv(recvBuffer, size01, 0, tag, comm, &requestReceive); // we into this buffer from rank 0

        while (!test(&requestSend))
        {
            // Busy wait
        }

        while (!test(&requestReceive))
        {
            // Busy wait
        }

        // Check the received value
        for (int i = 0; i < size01; i++)
            REQUIRE(recvBuffer[i] == sendValue01[i]);
    }
}
