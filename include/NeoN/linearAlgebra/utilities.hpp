// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 NeoN authors

#pragma once

#if NF_WITH_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/extensions/kokkos.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"


namespace NeoN::la
{

std::shared_ptr<gko::Executor> getGkoExecutor(Executor exec);

namespace detail
{

template<typename T>
gko::array<T> createGkoArray(std::shared_ptr<const gko::Executor> exec, std::span<T> values)
{
    return gko::make_array_view(exec, values.size(), values.data());
}

// template<typename T>
// gko::detail::const_array_view<T>
// createConstGkoArray(std::shared_ptr<const gko::Executor> exec, const std::span<const T> values)
// {
//     return gko::make_const_array_view(exec, values.size(), values.data());
// }

template<typename ValueType, typename IndexType>
std::shared_ptr<gko::matrix::Csr<ValueType, IndexType>> createGkoMtx(
    std::shared_ptr<const gko::Executor> exec, const LinearSystem<ValueType, IndexType>& sys
)
{
    auto nrows = static_cast<gko::dim<2>::dimension_type>(sys.rhs().size());
    auto mtx = sys.view().matrix;
    // NOTE we get a const view of the system but need a non const view to vals and indices
    // auto vals = createConstGkoArray(exec, mtx.values).copy_to_array();
    auto vals = gko::array<ValueType>::view(
        exec,
        static_cast<gko::size_type>(mtx.values.size()),
        const_cast<ValueType*>(mtx.values.data())
    );
    // auto col = createGkoArray(exec, mtx.colIdxs);
    auto col = gko::array<IndexType>::view(
        exec,
        static_cast<gko::size_type>(mtx.colIdxs.size()),
        const_cast<IndexType*>(mtx.colIdxs.data())
    );
    // auto row = createGkoArray(exec, mtx.rowOffs);
    auto row = gko::array<IndexType>::view(
        exec,
        static_cast<gko::size_type>(mtx.rowOffs.size()),
        const_cast<IndexType*>(mtx.rowOffs.data())
    );
    return gko::share(gko::matrix::Csr<ValueType, IndexType>::create(
        exec, gko::dim<2> {nrows, nrows}, vals, col, row
    ));
}

template<typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
createGkoDense(std::shared_ptr<const gko::Executor> exec, ValueType* ptr, localIdx s)
{
    auto size = static_cast<std::size_t>(s);
    return gko::share(gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2> {size, 1}, createGkoArray(exec, std::span {ptr, size}), 1
    ));
}

template<typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
createGkoDense(std::shared_ptr<const gko::Executor> exec, const ValueType* ptr, localIdx s)
{
    auto size = static_cast<std::size_t>(s);
    auto const_array_view = gko::array<ValueType>::const_view(exec, size, ptr);
    return gko::share(gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2> {size, 1}, const_array_view.copy_to_array(), 1
    ));
}

}

}

#endif
