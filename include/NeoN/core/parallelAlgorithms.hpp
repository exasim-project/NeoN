// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{


template<typename ValueType>
class Field;


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
    std::pair<size_t, size_t> range,
    Kernel kernel,
    std::string name = "parallelFor"
)
{
    auto [start, end] = range;
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        for (size_t i = start; i < end; i++)
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
            KOKKOS_LAMBDA(const size_t i) { kernel(i); }
        );
    }
}


template<parallelForKernel Kernel>
void parallelFor(
    const NeoFOAM::Executor& exec,
    std::pair<size_t, size_t> range,
    Kernel kernel,
    std::string name = "parallelFor"
)
{
    std::visit([&](const auto& e) { parallelFor(e, range, kernel, name); }, exec);
}

// Concept to check if a callable is compatible with ValueType(const size_t)
template<typename Kernel, typename ValueType>
concept parallelForFieldKernel = requires(Kernel t, ValueType val, size_t i) {
    {
        t(i)
    } -> std::same_as<ValueType>;
};

template<typename Executor, typename ValueType, parallelForFieldKernel<ValueType> Kernel>
void parallelFor(
    [[maybe_unused]] const Executor& exec,
    Field<ValueType>& field,
    Kernel kernel,
    std::string name = "parallelFor"
)
{
    auto view = field.view();
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        size_t fieldSize = field.size();
        for (size_t i = 0; i < fieldSize; i++)
        {
            view[i] = kernel(i);
        }
    }
    else
    {
        using runOn = typename Executor::exec;
        Kokkos::parallel_for(
            name,
            Kokkos::RangePolicy<runOn>(0, field.size()),
            KOKKOS_LAMBDA(const size_t i) { view[i] = kernel(i); }
        );
    }
}

template<typename ValueType, parallelForFieldKernel<ValueType> Kernel>
void parallelFor(Field<ValueType>& field, Kernel kernel, std::string name = "parallelFor")
{
    std::visit([&](const auto& e) { parallelFor(e, field, kernel, name); }, field.exec());
}

template<typename Executor, typename Kernel, typename T>
void parallelReduce(
    [[maybe_unused]] const Executor& exec, std::pair<size_t, size_t> range, Kernel kernel, T& value
)
{
    auto [start, end] = range;
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        for (size_t i = start; i < end; i++)
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
    const NeoFOAM::Executor& exec, std::pair<size_t, size_t> range, Kernel kernel, T& value
)
{
    std::visit([&](const auto& e) { parallelReduce(e, range, kernel, value); }, exec);
}


template<typename Executor, typename ValueType, typename Kernel, typename T>
void parallelReduce(
    [[maybe_unused]] const Executor& exec, Field<ValueType>& field, Kernel kernel, T& value
)
{
    if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
    {
        size_t fieldSize = field.size();
        for (size_t i = 0; i < fieldSize; i++)
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
void parallelReduce(Field<ValueType>& field, Kernel kernel, T& value)
{
    std::visit([&](const auto& e) { parallelReduce(e, field, kernel, value); }, field.exec());
}

template<typename Executor, typename Kernel>
void parallelScan(
    [[maybe_unused]] const Executor& exec, std::pair<size_t, size_t> range, Kernel kernel
)
{
    auto [start, end] = range;
    using runOn = typename Executor::exec;
    Kokkos::parallel_scan("parallelScan", Kokkos::RangePolicy<runOn>(start, end), kernel);
}

template<typename Kernel>
void parallelScan(const NeoFOAM::Executor& exec, std::pair<size_t, size_t> range, Kernel kernel)
{
    std::visit([&](const auto& e) { parallelScan(e, range, kernel); }, exec);
}

template<typename Executor, typename Kernel, typename ReturnType>
void parallelScan(
    [[maybe_unused]] const Executor& exec,
    std::pair<size_t, size_t> range,
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
    const NeoFOAM::Executor& exec,
    std::pair<size_t, size_t> range,
    Kernel kernel,
    ReturnType& returnValue
)
{
    std::visit([&](const auto& e) { parallelScan(e, range, kernel, returnValue); }, exec);
}


} // namespace NeoFOAM
