// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <complex>
#ifdef NF_WITH_MPI_SUPPORT
#include <mpi.h>
#endif
#include <type_traits>

#include "NeoFOAM/core/error.hpp"
#include "NeoFOAM/core/primitives/vector.hpp"

namespace NeoFOAM
{

#ifdef NF_WITH_MPI_SUPPORT

namespace mpi
{
/**
 * @brief Enumeration of MPI reduction operations
 */
enum class ReduceOp
{
    Max,
    Min,
    Sum,
    Prod,
    Land,
    Band,
    Lor,
    Bor,
    Maxloc,
    Minloc
};

/**
 * @brief Returns the corresponding MPI_Op for a given ReduceOp
 *
 * @param op The reduction operation
 * @return The corresponding MPI_Op
 */
constexpr MPI_Op getOp(const ReduceOp op)
{
    switch (op)
    {
    case ReduceOp::Max:
        return MPI_MAX;
    case ReduceOp::Min:
        return MPI_MIN;
    case ReduceOp::Sum:
        return MPI_SUM;
    case ReduceOp::Prod:
        return MPI_PROD;
    case ReduceOp::Land:
        return MPI_LAND;
    case ReduceOp::Band:
        return MPI_BAND;
    case ReduceOp::Lor:
        return MPI_LOR;
    case ReduceOp::Bor:
        return MPI_BOR;
    case ReduceOp::Maxloc:
        return MPI_MAXLOC;
    case ReduceOp::Minloc:
        return MPI_MINLOC;
    default:
        NF_ERROR_EXIT("Invalid MPI reduce operation requested.");
        return MPI_LOR; // This is to suppress the warning
    }
}

/**
 * @brief Returns the corresponding MPI_Datatype for a given C++ type
 *
 * @tparam valueType The C++ type
 * @return The corresponding MPI_Datatype
 */
template<typename valueType>
constexpr MPI_Datatype getType()
{
    if constexpr (std::is_same_v<valueType, char>) return MPI_CHAR;
    else if constexpr (std::is_same_v<valueType, wchar_t>)
        return MPI_WCHAR;
    else if constexpr (std::is_same_v<valueType, short>)
        return MPI_SHORT;
    else if constexpr (std::is_same_v<valueType, int>)
        return MPI_INT;
    else if constexpr (std::is_same_v<valueType, long>)
        return MPI_LONG;
    else if constexpr (std::is_same_v<valueType, long long>)
        return MPI_LONG_LONG;
    else if constexpr (std::is_same_v<valueType, unsigned short>)
        return MPI_UNSIGNED_SHORT;
    else if constexpr (std::is_same_v<valueType, unsigned>)
        return MPI_UNSIGNED;
    else if constexpr (std::is_same_v<valueType, unsigned long>)
        return MPI_UNSIGNED_LONG;
    else if constexpr (std::is_same_v<valueType, unsigned long long>)
        return MPI_UNSIGNED_LONG_LONG;
    else if constexpr (std::is_same_v<valueType, float>)
        return MPI_FLOAT;
    else if constexpr (std::is_same_v<valueType, double>)
        return MPI_DOUBLE;
    else if constexpr (std::is_same_v<valueType, long double>)
        return MPI_LONG_DOUBLE;
    else if constexpr (std::is_same_v<valueType, bool>)
        return MPI_CXX_BOOL;
    else if constexpr (std::is_same_v<valueType, std::complex<float>>)
        return MPI_CXX_FLOAT_COMPLEX;
    else if constexpr (std::is_same_v<valueType, std::complex<double>>)
        return MPI_CXX_DOUBLE_COMPLEX;
    else if constexpr (std::is_same_v<valueType, std::complex<long double>>)
        return MPI_CXX_LONG_DOUBLE_COMPLEX;
    else
        NF_ERROR_EXIT("Invalid MPI datatype requested.");
    return MPI_CHAR; // This is to suppress the warning
}

/**
 * @brief Performs a blocking all-reduce operation on a value across all processes in the
 * communicator.
 *
 * @tparam valueType The type of the value.
 * @param value The value to be all-reduced.
 * @param op The reduction operation to be performed.
 * @param comm The communicator across which the reduction operation is performed.
 * @note Blocking MPI operation.
 */
template<typename valueType>
void allReduce(valueType& value, const ReduceOp op, MPI_Comm comm)
{
    MPI_Allreduce(
        MPI_IN_PLACE, reinterpret_cast<void*>(&value), 1, getType<valueType>(), getOp(op), comm
    );
}

/**
 * @brief Performs a blocking all-reduce operation on a vector across all processes in the
 * communicator.
 *
 * @param vector The vector to be all-reduced.
 * @param op The reduction operation to be performed.
 * @param comm The communicator across which the reduction operation is performed.
 * @note Blocking MPI operation.
 */
template<>
inline void allReduce(Vector& vector, const ReduceOp op, MPI_Comm comm)
{
    MPI_Allreduce(
        MPI_IN_PLACE,
        reinterpret_cast<void*>(vector.data()),
        static_cast<mpi_label_t>(vector.size()),
        getType<scalar>(),
        getOp(op),
        comm
    );
}

/**
 * @brief Non-blocking send of a set of scalar values to a remote rank.
 *
 * @tparam valueType The type of the scalar value.
 * @param buffer Pointer to first scalar value to be sent.
 * @param size The size of the send buffer, i.e. number of components/elements.
 * @param rankReceive The receiving rank index.
 * @param tag The tag of the message, used to identify the communication.
 * @param comm The MPI communicator across which the message is sent.
 * @param request Pointer to the MPI_Request object, is populated by the function.
 * @note Non-blocking MPI operation.
 */
template<typename valueType>
void isend(
    const valueType* buffer,
    const mpi_label_t size,
    mpi_label_t rankReceive,
    mpi_label_t tag,
    MPI_Comm comm,
    MPI_Request* request
)
{
    mpi_label_t err =
        MPI_Isend(buffer, size, getType<valueType>(), rankReceive, tag, comm, request);
    NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Isend failed.");
}

/**
 * @brief Non-blocking receive of a set of scalar values from a remote rank.
 *
 * @tparam valueType The type of the scalar value.
 * @param buffer Pointer to the buffer where the received scalar values will be stored.
 * @param size The size of the receive buffer, i.e. number of components/elements.
 * @param rankSend The rank index of the sender.
 * @param tag The tag of the message, used to identify the communication.
 * @param comm The MPI communicator across which the message is received.
 * @param request Pointer to the MPI_Request object, is populated by the function.
 * @note Non-blocking MPI operation.
 */
template<typename valueType>
void irecv(
    valueType* buffer,
    const mpi_label_t size,
    mpi_label_t rankSend,
    mpi_label_t tag,
    MPI_Comm comm,
    MPI_Request* request
)
{
    mpi_label_t err = MPI_Irecv(buffer, size, getType<valueType>(), rankSend, tag, comm, request);
    NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Irecv failed.");
}

/**
 * @brief Tests if a non-blocking communication request has completed.
 *
 * @param request Pointer to the MPI_Request object.
 * @return True if the request has completed, false otherwise.
 * @note Non-blocking MPI operation.
 */
inline bool test(MPI_Request* request)
{
    mpi_label_t flag;
    mpi_label_t err = MPI_Test(request, &flag, MPI_STATUS_IGNORE);
    NF_DEBUG_ASSERT(err == MPI_SUCCESS, "MPI_Test failed.");
    return static_cast<bool>(flag);
}

} // namespace mpi

#endif

}
