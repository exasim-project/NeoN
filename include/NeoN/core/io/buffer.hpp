// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#pragma once

#include <memory>

namespace NeoN::io
{

/*
 * @class Buffer
 * @brief Type-erased interface to store the IO buffer like adios2::Variable.
 *
 * The Buffer class is a type-erased interface to store buffer components from IO
 * libraries wrapped into classes that cohere with the Buffer's interface.
 */
class Buffer
{
public:

    /*
     * @brief Constructs an Buffer component for IO from a specific BufferType.
     *
     * @tparam BufferType The buffer type that wraps a specific IO library.
     * @param buffer The buffer instance to be wrapped.
     */
    template<typename BufferType>
    Buffer(BufferType buffer) : pimpl_(std::make_unique<BufferModel<BufferType>>(std::move(buffer)))
    {}

private:

    /*
     * @brief Base concept declaring the type-erased interface
     */
    struct BufferConcept
    {
        virtual ~BufferConcept() = default;
    };

    /*
     * @brief Derived model delegating the implementation to the type-erased PIMPL
     */
    template<typename BufferType>
    struct BufferModel : BufferConcept
    {
        BufferModel(BufferType buffer) : buffer_(std::move(buffer_)) {}
        BufferType buffer_;
    };

    /*
     * @brief Type-erased PIMPL
     */
    std::unique_ptr<BufferConcept> pimpl_;
};

} // namespace NeoN::io
