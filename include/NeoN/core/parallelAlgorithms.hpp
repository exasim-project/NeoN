// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "NeoN/core/logging.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/executor/executor.hpp"

#ifdef NN_WITH_KOKKOS
#define NEON_LAMBDA KOKKOS_LAMBDA
#define NEON_INLINE_FUNCTION KOKKOS_INLINE_FUNCTION
namespace NeoN
{
// just pull Kokkos::atomic_* functions into NeoN namespace
using Kokkos::atomic_add;
using Kokkos::atomic_sub;
}
#else
#define NEON_LAMBDA [&]
namespace NeoN
{
// using atomic_add = [](auto& a, auto b){a+b;};
// using atomic_sub = [](auto& a, auto b){a-b;};
}
#endif

namespace NeoN
{


template<typename ValueType>
class Vector;


// Concept to check if a callable is compatible with void(const size_t)
template<typename Kernel>
concept parallelForKernel = requires(Kernel t, size_t i) {
    {
        t(i)
    } -> std::same_as<void>;
};


/* @brief calls fence if a logger is set */
template<typename ExecutorType>
void fenceIfLogger(const ExecutorType& exec)
{
    auto logger = getLogger(exec);
    if (logger != nullptr)
    {
        fence(exec);
    }
}

/* @brief execute parallelFor with concrete executor */
template<typename ExecutorType, parallelForKernel Kernel>
void parallelFor(
    const ExecutorType&, std::pair<localIdx, localIdx> range, Kernel kernel, std::string name
)
{
    auto [start, end] = range;

    if constexpr (std::is_same<std::remove_reference_t<ExecutorType>, SerialExecutor>::value)
    {
        for (localIdx i = start; i < end; i++)
        {
            kernel(i);
        }
    }
    else
    {
        using runOn = typename ExecutorType::exec;
        Kokkos::parallel_for(
            name,
            Kokkos::RangePolicy<runOn>(start, end),
            NEON_LAMBDA(const localIdx i) { kernel(i); }
        );
    }
}


/* @brief dispatch parallelFor based on executor variant type */
template<parallelForKernel Kernel>
void parallelFor(
    const NeoN::Executor& exec,
    std::pair<localIdx, localIdx> range,
    Kernel kernel,
    std::string name = "parallelFor"
)
{
    std::visit([&](const auto& e) { parallelFor(e, range, kernel, name); }, exec);
}

// Concept to check if a callable is compatible with ValueType(const size_t)
template<typename Kernel, typename ValueType>
concept parallelForContainerKernel = requires(Kernel t, ValueType val, size_t i) {
    {
        t(i)
    } -> std::same_as<ValueType>;
};

template<
    typename Executor,
    template<typename>
    class ContType,
    typename ValueType,
    parallelForContainerKernel<ValueType> Kernel>
void parallelFor(
    const Executor&, ContType<ValueType>& container, Kernel kernel, std::string name = "parallelFor"
)
{
    auto view = container.view();
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        for (localIdx i = 0; i < view.size(); i++)
        {
            view[i] = kernel(i);
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_for(
            name,
            Kokkos::RangePolicy<runOn>(0, view.size()),
            NEON_LAMBDA(const localIdx i) { view[i] = kernel(i); }
        );
    }
}

template<
    template<typename>
    class ContType,
    typename ValueType,
    parallelForContainerKernel<ValueType> Kernel>
void parallelFor(ContType<ValueType>& cont, Kernel kernel, std::string name = "parallelFor")
{
    std::visit([&](const auto& e) { parallelFor(e, cont, kernel, name); }, cont.exec());
}

template<typename Executor, typename Kernel, typename T>
void parallelReduce(
    [[maybe_unused]] const Executor& exec,
    std::pair<localIdx, localIdx> range,
    Kernel kernel,
    T& value
)
{
    auto [start, end] = range;
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        for (localIdx i = start; i < end; i++)
        {
            if constexpr (Kokkos::is_reducer<T>::value)
            {
                kernel(i, value.reference());
            }
            else
            {
                kernel(i, value);
            }
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_reduce(
            "parallelReduce", Kokkos::RangePolicy<runOn>(start, end), kernel, value
        );
    }
}

template<typename Kernel, typename T>
void parallelReduce(
    const NeoN::Executor& exec, std::pair<localIdx, localIdx> range, Kernel kernel, T& value
)
{
    std::visit([&](const auto& e) { parallelReduce(e, range, kernel, value); }, exec);
}


template<typename Executor, typename ValueType, typename Kernel, typename T>
void parallelReduce(
    [[maybe_unused]] const Executor& exec, Vector<ValueType>& field, Kernel kernel, T& value
)
{
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        localIdx fieldSize = field.size();
        for (localIdx i = 0; i < fieldSize; i++)
        {
            if constexpr (Kokkos::is_reducer<T>::value)
            {
                kernel(i, value.reference());
            }
            else
            {
                kernel(i, value);
            }
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_reduce(
            "parallelReduce", Kokkos::RangePolicy<runOn>(0, field.size()), kernel, value
        );
    }
}

template<typename ValueType, typename Kernel, typename T>
void parallelReduce(Vector<ValueType>& field, Kernel kernel, T& value)
{
    std::visit([&](const auto& e) { parallelReduce(e, field, kernel, value); }, field.exec());
}

// Deduce the scan accumulator type (the second, by-reference, parameter) from a
// parallel_scan kernel of the form void(localIdx, Accumulator&, bool). Needed
// both by the SerialExecutor branch below (which threads the accumulator itself)
// and to spell the signature of the forwarding lambda handed to Kokkos.
namespace detail
{
template<typename Kernel>
struct ScanAccumulator
{
    template<typename C, typename R, typename I, typename U>
    static U deduce(R (C::*)(I, U&, bool) const);

    template<typename C, typename R, typename I, typename U>
    static U deduce(R (C::*)(I, U&, bool));

    using type = decltype(deduce(&Kernel::operator()));
};
}

// NOTE: the kernel is taken by const-reference (not by value) all the way down
// the dispatch chain. When `parallelScan` is reached through the Executor-variant
// overloads below, std::visit would otherwise copy the kernel on the host once per
// hop. For an nvcc extended lambda (NEON_LAMBDA) compiled in a translation unit
// that contains no direct device launch of that lambda type, nvcc never emits the
// lambda's host trampolines (fp_caller/fp_copier/fp_deleter), so such a host-side
// copy dereferences a null fp_copier and segfaults. Passing by reference means the
// SerialExecutor path only *calls* the kernel and never copies it.
template<typename Executor, typename Kernel>
void parallelScan(
    [[maybe_unused]] const Executor& exec, std::pair<localIdx, localIdx> range, const Kernel& kernel
)
{
    auto [start, end] = range;
    using Accumulator = typename detail::ScanAccumulator<Kernel>::type;
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        // Do not dispatch the (nvcc extended) lambda to Kokkos on the serial
        // backend; emulate the inclusive scan with a plain host loop instead.
        Accumulator update {};
        for (localIdx i = start; i < end; i++)
        {
            kernel(i, update, true);
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_scan(
            "parallelScan",
            Kokkos::RangePolicy<runOn>(start, end),
            NEON_LAMBDA(const localIdx i, Accumulator& update, const bool final) {
                kernel(i, update, final);
            }
        );
    }
}

template<typename Kernel>
void parallelScan(
    const NeoN::Executor& exec, std::pair<localIdx, localIdx> range, const Kernel& kernel
)
{
    std::visit([&](const auto& e) { parallelScan(e, range, kernel); }, exec);
}

template<typename Executor, typename Kernel, typename ReturnType>
void parallelScan(
    [[maybe_unused]] const Executor& exec,
    std::pair<localIdx, localIdx> range,
    const Kernel& kernel,
    ReturnType& returnValue
)
{
    auto [start, end] = range;
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        // Do not dispatch the (nvcc extended) lambda to Kokkos on the serial
        // backend; emulate the inclusive scan with a plain host loop instead.
        ReturnType update {};
        for (localIdx i = start; i < end; i++)
        {
            kernel(i, update, true);
        }
        returnValue = update;
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_scan(
            "parallelScan",
            Kokkos::RangePolicy<runOn>(start, end),
            NEON_LAMBDA(const localIdx i, ReturnType& update, const bool final) {
                kernel(i, update, final);
            },
            returnValue
        );
    }
}

template<typename Kernel, typename ReturnType>
void parallelScan(
    const NeoN::Executor& exec,
    std::pair<localIdx, localIdx> range,
    const Kernel& kernel,
    ReturnType& returnValue
)
{
    std::visit([&](const auto& e) { parallelScan(e, range, kernel, returnValue); }, exec);
}

} // namespace NeoN
