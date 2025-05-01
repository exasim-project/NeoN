// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/executor/executor.hpp"

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

template<typename Executor, parallelForKernel Kernel>
void parallelFor(
    [[maybe_unused]] const Executor& exec,
    std::pair<localIdx, localIdx> range,
    Kernel kernel,
    std::string name = "parallelFor"
)
{
    auto [start, end] = range;
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        for (localIdx i = start; i < end; i++)
        {
            kernel(i);
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_for(
            name,
            Kokkos::RangePolicy<runOn>(start, end),
            KOKKOS_LAMBDA(const localIdx i) { kernel(i); }
        );
    }
}


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
    [[maybe_unused]] const Executor& exec,
    ContType<ValueType>& container,
    Kernel kernel,
    std::string name = "parallelFor"
)
{
    auto view = container.view();
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        for (localIdx i = 0; i < container.size(); i++)
        {
            view[i] = kernel(i);
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_for(
            name,
            Kokkos::RangePolicy<runOn>(0, container.size()),
            KOKKOS_LAMBDA(const localIdx i) { view[i] = kernel(i); }
        );
    }
}

template<
    template<typename>
    class ContType,
    typename ValueType,
    parallelForContainerKernel<ValueType> Kernel>
void parallelFor(ContType<ValueType>& field, Kernel kernel, std::string name = "parallelFor")
{
    std::visit([&](const auto& e) { parallelFor(e, field, kernel, name); }, field.exec());
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

template<typename Executor, typename Kernel>
void parallelScan(
    [[maybe_unused]] const Executor& exec, std::pair<localIdx, localIdx> range, Kernel kernel
)
{
    auto [start, end] = range;
    using runOn = typename Executor::exec;
    Kokkos::parallel_scan("parallelScan", Kokkos::RangePolicy<runOn>(start, end), kernel);
}

template<typename Kernel>
void parallelScan(const NeoN::Executor& exec, std::pair<localIdx, localIdx> range, Kernel kernel)
{
    std::visit([&](const auto& e) { parallelScan(e, range, kernel); }, exec);
}

template<typename Executor, typename Kernel, typename ReturnType>
void parallelScan(
    [[maybe_unused]] const Executor& exec,
    std::pair<localIdx, localIdx> range,
    Kernel kernel,
    ReturnType& returnValue
)
{
    auto [start, end] = range;
    using runOn = typename Executor::exec;
    Kokkos::parallel_scan(
        "parallelScan", Kokkos::RangePolicy<runOn>(start, end), kernel, returnValue
    );
}

template<typename Kernel, typename ReturnType>
void parallelScan(
    const NeoN::Executor& exec,
    std::pair<localIdx, localIdx> range,
    Kernel kernel,
    ReturnType& returnValue
)
{
    std::visit([&](const auto& e) { parallelScan(e, range, kernel, returnValue); }, exec);
}


} // namespace NeoN
