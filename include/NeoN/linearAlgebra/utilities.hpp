// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 NeoFOAM authors

#pragma once

#if NF_WITH_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"


namespace NeoFOAM::la
{

std::shared_ptr<gko::Executor> getGkoExecutor(Executor exec);

namespace detail
{

template<typename T>
gko::array<T> createGkoArray(std::shared_ptr<const gko::Executor> exec, std::span<T> values)
{
    return gko::make_array_view(exec, values.size(), values.data());
}

template<typename T>
gko::detail::const_array_view<T>
createConstGkoArray(std::shared_ptr<const gko::Executor> exec, const std::span<const T> values)
{
    return gko::make_const_array_view(exec, values.size(), values.data());
}

template<typename ValueType, typename IndexType>
std::shared_ptr<gko::matrix::Csr<ValueType, int>> createGkoMtx(
    std::shared_ptr<const gko::Executor> exec, const LinearSystem<ValueType, IndexType>& sys
)
{
    size_t nrows = sys.rhs().size();
    // TODO: avoid copying
    // NOTE: since mtx is a converted copy we need to make sure that mtx data is not
    // deallocated before solving
    auto mtx = convert<ValueType, IndexType, ValueType, int>(sys.exec(), sys.view().matrix);
    auto vals = createGkoArray(exec, mtx.values().view());
    auto col = createGkoArray(exec, mtx.colIdxs().view());
    auto row = createGkoArray(exec, mtx.rowPtrs().view());
    return gko::share(
        gko::matrix::Csr<ValueType, int>::create(exec, gko::dim<2> {nrows, nrows}, vals, col, row)
    );
}

template<typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
createGkoDense(std::shared_ptr<const gko::Executor> exec, ValueType* ptr, size_t size)
{
    return gko::share(gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2> {size, 1}, createGkoArray(exec, std::span {ptr, size}), 1
    ));
}

template<typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
createGkoDense(std::shared_ptr<const gko::Executor> exec, const ValueType* ptr, size_t size)
{
    auto const_array_view = gko::array<ValueType>::const_view(exec, size, ptr);
    return gko::share(gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2> {size, 1}, const_array_view.copy_to_array(), 1
    ));
}

}

}

#endif
