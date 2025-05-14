// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <type_traits>
#include <tuple>
#include <Kokkos_Core.hpp>

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/view.hpp"

namespace NeoN
{


namespace detail
{

/**
 * @brief A helper function to simplify the common pattern of copying between and to executor.
 * @param size The number of elements to copy.
 * @param srcPtr Pointer to the original block of memory.
 * @param dstPtr Pointer to the target block of memory.
 * @tparam ValueType The type of the underlying elements.
 * @returns A function that takes a source and an destination executor
 */
template<typename ValueType>
auto deepCopyVisitor(localIdx ssize, const ValueType* srcPtr, ValueType* dstPtr)
{
    size_t size = static_cast<size_t>(ssize);
    return [size, srcPtr, dstPtr](const auto& srcExec, const auto& dstExec)
    {
        Kokkos::deep_copy(
            dstExec.createKokkosView(dstPtr, size), srcExec.createKokkosView(srcPtr, size)
        );
    };
};

}


/**
 * @brief Map a field using a specific executor.
 *
 * @param cont The container to map.
 * @param inner The function to apply to each element of the field.
 * @param range The range to map the field in. If not provided, the whole field is mapped.
 */
template<template<typename> class ContType, typename ValueType, typename Inner>
void map(ContType<ValueType>& cont, const Inner inner, std::pair<localIdx, localIdx> range = {0, 0})
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = cont.size();
    }
    auto contView = cont.view();
    parallelFor(
        cont.exec(), {start, end}, KOKKOS_LAMBDA(const localIdx i) { contView[i] = inner(i); }
    );
}


/**
 * @brief Fill the field with a vector value using a specific executor.
 *
 * @param field The field to fill.
 * @param value The vector value to fill the field with.
 * @param range The range to fill the field in. If not provided, the whole field is filled.
 */
template<template<typename> class ContType, typename ValueType>
void fill(
    ContType<ValueType>& cont,
    const std::type_identity_t<ValueType> value,
    std::pair<localIdx, localIdx> range = {0, 0}
)
{
    auto [start, end] = range;
    NF_DEBUG_ASSERT(start <= end, "Range must be ordered in ascending fashion");
    if (end == 0)
    {
        end = cont.size();
    }
    auto viewA = cont.view();
    parallelFor(
        cont.exec(), {start, end}, KOKKOS_LAMBDA(const localIdx i) { viewA[i] = value; }
    );
}


/**
 * @brief Set the container with a view of values using a specific executor.
 *
 * @param cont The container to set.
 * @param view The view of values to set the container with.
 * @param range The range to set the container in. If not provided, the whole container is set.
 */
template<template<typename> class ContType, typename ValueType>
void setContainer(
    ContType<ValueType>& cont,
    const View<const std::type_identity_t<ValueType>> view,
    std::pair<localIdx, localIdx> range = {0, 0}
)
{
    auto [start, end] = range;
    if (end == 0)
    {
        end = cont.size();
    }
    auto contView = cont.view();
    parallelFor(
        cont.exec(), {start, end}, KOKKOS_LAMBDA(const localIdx i) { contView[i] = view[i]; }
    );
}

template<typename... Args>
auto copyToHosts(Args&... cont)
{
    return std::make_tuple(cont.copyToHost()...);
}

template<template<typename> class ContType, typename ValueType>
bool equal(ContType<ValueType>& cont, ValueType value)
{
    auto hostCont = cont.copyToHost();
    auto hostView = hostCont.view();
    for (localIdx i = 0; i < hostView.size(); i++)
    {
        if (hostView[i] != value)
        {
            return false;
        }
    }
    return true;
};

template<template<typename> class ContType, typename ValueType>
bool equal(const ContType<ValueType>& cont1, const ContType<ValueType>& cont2)
{
    auto [hostCont1, hostCont2] = copyToHosts(cont1, cont2);
    auto [hostView1, hostView2] = views(hostCont1, hostCont2);

    if (hostView1.size() != hostView2.size())
    {
        return false;
    }

    for (localIdx i = 0; i < hostView1.size(); i++)
    {
        if (hostView1[i] != hostView2[i])
        {
            return false;
        }
    }

    return true;
};

template<template<typename> class ContType, typename ValueType>
bool equal(const ContType<ValueType>& cont, View<ValueType> view2)
{
    auto hostView = cont.copyToHost().view();

    if (hostView.size() != view2.size())
    {
        return false;
    }

    for (localIdx i = 0; i < hostView.size(); i++)
    {
        if (hostView[i] != view2[i])
        {
            return false;
        }
    }

    return true;
}

}
