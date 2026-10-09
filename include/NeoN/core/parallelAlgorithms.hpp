// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#if defined(NEON_HOST_ONLY_TU)
#error "parallelAlgorithms.hpp launches Kokkos kernels and must not be included from a host-only \
translation unit. Move the kernel into a device TU and call it through a non-template function."
#endif

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "NeoN/core/logging.hpp"
#include "NeoN/core/portability.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/executor/kokkosExecutor.hpp"

// NeoN_public_api always defines NN_WITH_KOKKOS (to 1 or 0), so this has to test the value, not
// whether the macro is defined. NeoN has no non-Kokkos kernel backend.
#if !NN_WITH_KOKKOS
#error "NeoN kernels require Kokkos (NeoN_WITH_KOKKOS=ON)"
#endif

#define NEON_LAMBDA KOKKOS_LAMBDA

namespace NeoN
{
// just pull Kokkos::atomic_* functions into NeoN namespace
using Kokkos::atomic_add;
using Kokkos::atomic_sub;
}

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


namespace detail
{
/* @brief call f with the concrete executor held by exec
 *
 * Deliberately not std::visit: its dispatch table is a GNU_UNIQUE object the dynamic linker
 * merges process-wide, so a table instantiated in another DSO can route a kernel built here into
 * that DSO's copy of the code. For nvcc extended lambdas (NEON_LAMBDA) the call slot is TU-local,
 * so the foreign copy calls 0x0 (exasim-project/NeoFOAM#410). Direct calls stay in the calling
 * DSO when it is linked with -Bsymbolic-functions.
 */
template<typename F>
void visitExecutor(const Executor& exec, F&& f)
{
    if (const auto* serial = std::get_if<SerialExecutor>(&exec))
    {
        f(*serial);
    }
    else if (const auto* cpu = std::get_if<CPUExecutor>(&exec))
    {
        f(*cpu);
    }
    else
    {
        f(std::get<GPUExecutor>(exec));
    }
}
}

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
        using runOn = kokkosExecSpace<ExecutorType>;
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
    detail::visitExecutor(exec, [&](const auto& e) { parallelFor(e, range, kernel, name); });
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
        using runOn = kokkosExecSpace<Executor>;
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
    detail::visitExecutor(cont.exec(), [&](const auto& e) { parallelFor(e, cont, kernel, name); });
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
        using runOn = kokkosExecSpace<Executor>;
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
    detail::visitExecutor(exec, [&](const auto& e) { parallelReduce(e, range, kernel, value); });
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
        using runOn = kokkosExecSpace<Executor>;
        Kokkos::parallel_reduce(
            "parallelReduce", Kokkos::RangePolicy<runOn>(0, field.size()), kernel, value
        );
    }
}

template<typename ValueType, typename Kernel, typename T>
void parallelReduce(Vector<ValueType>& field, const Kernel& kernel, T& value)
{
    detail::visitExecutor(
        field.exec(), [&](const auto& e) { parallelReduce(e, field, kernel, value); }
    );
}

template<typename Executor, typename Kernel>
void parallelScan(
    [[maybe_unused]] const Executor& exec, std::pair<localIdx, localIdx> range, const Kernel& kernel
)
{
    auto [start, end] = range;
    using runOn = kokkosExecSpace<Executor>;
    Kokkos::parallel_scan("parallelScan", Kokkos::RangePolicy<runOn>(start, end), kernel);
}

template<typename Kernel>
void parallelScan(
    const NeoN::Executor& exec, std::pair<localIdx, localIdx> range, const Kernel& kernel
)
{
    detail::visitExecutor(exec, [&](const auto& e) { parallelScan(e, range, kernel); });
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
    using runOn = kokkosExecSpace<Executor>;
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
    detail::visitExecutor(
        exec, [&](const auto& e) { parallelScan(e, range, kernel, returnValue); }
    );
}

} // namespace NeoN
