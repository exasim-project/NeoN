// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include <Kokkos_Core.hpp>
#include "NeoFOAM/core/executor/executor.hpp"

namespace NeoFOAM
{

struct SumKernel
{
    template<typename T>
    void operator()([[maybe_unused]] const GPUExecutor& exec, const Field<T>& field, T& sum) const
    {
        using executor = typename GPUExecutor::exec;
        auto fieldS = field.view();
        auto end = field.size();
        Kokkos::parallel_reduce(
            "sum",
            Kokkos::RangePolicy<executor>(0, end),
            KOKKOS_LAMBDA(const size_t i, T& lsum) { lsum += fieldS[i]; },
            sum
        );
    }

    template<typename T>
    void operator()([[maybe_unused]] const CPUExecutor& exec, const Field<T>& field, T& sum)
    {
        using executor = typename CPUExecutor::exec;
        auto fieldS = field.view();
        auto end = field.size();
        Kokkos::parallel_reduce(
            "sum",
            Kokkos::RangePolicy<executor>(0, end),
            KOKKOS_LAMBDA(const size_t i, T& lsum) { lsum += fieldS[i]; },
            sum
        );
    }

    template<typename T>
    void operator()([[maybe_unused]] const SerialExecutor& exec, const Field<T>& field, T& sum)
    {
        using executor = typename SerialExecutor::exec;
        auto fieldS = field.view();
        auto end = field.size();
        Kokkos::parallel_reduce(
            "sum",
            Kokkos::RangePolicy<executor>(0, end),
            KOKKOS_LAMBDA(const size_t i, T& lsum) { lsum += fieldS[i]; },
            sum
        );
    }
};


/*template<typename T>*/
/*T sum(const Field<T>& field)*/
/*{*/
/*    T sumValue {};*/
/*    SumKernel kernel {};*/
/*    std::visit([&](const auto& exec) { kernel(exec, field, sumValue); }, field.exec());*/
/*    return sumValue;*/
/*};*/
/**/
/*template<>*/
/*scalar sum(const Field<scalar>& field)*/
/*{*/
/*    scalar sumValue = 0;*/
/*    SumKernel kernel {};*/
/*    std::visit([&](const auto& exec) { kernel(exec, field, sumValue); }, field.exec());*/
/*    return sumValue;*/
/*};*/


} // namespace NeoFOAM
