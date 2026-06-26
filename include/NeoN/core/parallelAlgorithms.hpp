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
    const ExecutorType&, std::pair<localIdx, localIdx> range, const Kernel& kernel, std::string name
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
        // Pass `kernel` DIRECTLY to Kokkos (do not wrap it in another NEON_LAMBDA). The wrapper
        // makes `kernel`'s type only ever HOST-copied (into the wrapper's closure) and never the
        // type actually device-launched, so nvcc emits no — or, under -O3, a NULL — host copy
        // trampoline (__nv_hdl_wrapper_t::do_copy) for it; the host-side copy then jumps to 0x0
        // and SIGSEGVs. Launching `kernel` directly makes it the device-launched type, forcing
        // nvcc to emit a correct trampoline. See neon-nvcc-extended-lambda-trampoline.
        Kokkos::parallel_for(
            name, Kokkos::RangePolicy<runOn, Kokkos::IndexType<localIdx>>(start, end), kernel
        );
    }
}


/* @brief dispatch parallelFor based on executor variant type */
template<parallelForKernel Kernel>
void parallelFor(
    const NeoN::Executor& exec,
    std::pair<localIdx, localIdx> range,
    const Kernel& kernel,
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
    const Executor&,
    ContType<ValueType>& container,
    const Kernel& kernel,
    std::string name = "parallelFor"
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
void parallelFor(ContType<ValueType>& cont, const Kernel& kernel, std::string name = "parallelFor")
{
    std::visit([&](const auto& e) { parallelFor(e, cont, kernel, name); }, cont.exec());
}

template<typename Executor, typename Kernel, typename T>
void parallelReduce(
    [[maybe_unused]] const Executor& exec,
    std::pair<localIdx, localIdx> range,
    const Kernel& kernel,
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
    const NeoN::Executor& exec, std::pair<localIdx, localIdx> range, const Kernel& kernel, T& value
)
{
    std::visit([&](const auto& e) { parallelReduce(e, range, kernel, value); }, exec);
}


template<typename Executor, typename ValueType, typename Kernel, typename T>
void parallelReduce(
    [[maybe_unused]] const Executor& exec, Vector<ValueType>& field, const Kernel& kernel, T& value
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
void parallelReduce(Vector<ValueType>& field, const Kernel& kernel, T& value)
{
    std::visit([&](const auto& e) { parallelReduce(e, field, kernel, value); }, field.exec());
}

template<typename Executor, typename Kernel>
void parallelScan(
    [[maybe_unused]] const Executor& exec, std::pair<localIdx, localIdx> range, const Kernel& kernel
)
{
    auto [start, end] = range;
    using runOn = typename Executor::exec;
    Kokkos::parallel_scan("parallelScan", Kokkos::RangePolicy<runOn>(start, end), kernel);
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
    using runOn = typename Executor::exec;
    Kokkos::parallel_scan(
        "parallelScan", Kokkos::RangePolicy<runOn>(start, end), kernel, returnValue
    );
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
