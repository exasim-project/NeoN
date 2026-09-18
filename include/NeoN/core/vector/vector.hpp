// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/error.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/view.hpp"

#include <vector>


namespace NeoN
{
/**
 * @class Vector
 * @brief An executor-aware contiguous container for numerical data with vector-style arithmetic
 * semantics.
 *
 * @details
 * Vector is intended for numerical quantities for which arithmetic operations
 * on the elements or on the vector as a whole are meaningful.
 *
 * Vector provides operations for performing arithmetic on its stored values.
 * Use Vector when the numerical meaning of the data and its arithmetic operations
 * are an important part of its interface.
 *
 * For structural, indexing, or other non-arithmetic data, use Array instead.
 *
 * @tparam ValueType The type of the elements stored in the vector.
 *
 * @ingroup Vectors
 */
template<typename ValueType>
class Vector
{

public:

    using VectorValueType = ValueType;

    /**
     * @brief Creates an uninitialized Vector with the given size on an executor.
     *
     * @param exec Executor on which the vector data is allocated.
     * @param size Number of elements in the vector.
     */
    Vector(const Executor& exec, localIdx size);

    /**
     * @brief Creates a Vector with the given size from existing data on an executor.
     *
     * @param exec Executor on which the vector data is allocated.
     * @param in Pointer to the source data.
     * @param size Number of elements to copy.
     * @param hostExec Executor on which the source data is located.
     */
    Vector(
        const Executor& exec,
        const ValueType* in,
        localIdx size,
        Executor hostExec = SerialExecutor()
    );

    /**
     * @brief Creates a Vector with the given size and initializes all elements to a value.
     *
     * @param exec Executor on which the Vector data is allocated.
     * @param size Number of elements in the Vector.
     * @param value Value used to initialize every element.
     */
    Vector(const Executor& exec, localIdx size, ValueType value);

    /**
     * @brief Creates a Vector from a std::vector on the given executor.
     *
     * @param exec Executor on which the Vector data is allocated.
     * @param in std::vector containing the values to copy into the Vector.
     */
    Vector(const Executor& exec, std::vector<ValueType> in);

    /**
     * @brief Creates a copy of another Vector on the specified executor.
     *
     * @param exec Executor on which the new Vector is allocated.
     * @param in Vector to copy.
     */
    Vector(const Executor& exec, const Vector<ValueType>& in);

    /**
     * @brief Copy constructor.
     *
     * @param rhs Vector to copy from.
     */
    Vector(const Vector<ValueType>& rhs);

    /**
     * @brief Move constructor.
     *
     * @param rhs Vector whose resources are transferred to the new Vector.
     */
    Vector(Vector<ValueType>&& rhs) noexcept;

    /**
     * @brief Destroys the Vector object.
     */
    ~Vector();

    /**
     * @brief Applies a function to the elements of the Vector.
     *
     * @param f Function or functor applied to the Vector elements.
     *
     * @note The function must be compatible with execution by Kokkos. Ideally, it should be a
     * Kokkos lambda.
     */
    template<typename func>
    void apply(func f)
    {
        map(*this, f);
    }

    /**
     * @brief Creates a copy of the Vector on the specified executor.
     *
     * @param dstExec Executor on which the copied Vector is allocated.
     * @return A copy of this Vector on @p dstExec.
     */
    [[nodiscard]] Vector<ValueType> copyToExecutor(Executor dstExec) const;

    /**
     * @brief Creates a host-resident copy of the Vector.
     *
     * @return A copy of this Vector allocated on the host executor.
     */
    [[nodiscard]] Vector<ValueType> copyToHost() const;

    /**
     * @brief Copies the Vector data to a host-resident Vector.
     *
     * @param result Destination Vector. Its size must match this Vector.
     *
     * @warning An error is raised if @p result does not have the same size as
     * this Vector.
     */
    void copyToHost(Vector<ValueType>& result);

    /**
     * @brief Host-side element access is intentionally disabled.
     *
     * Vector data may reside in executor-specific memory, such as GPU memory.
     * Direct host-side element access could therefore result in invalid memory
     * access.
     *
     * @note Use @ref view() or an executor-aware operation to access the data.
     */
    ValueType& operator[](const localIdx i) = delete;

    const ValueType& operator[](const localIdx i) const = delete;

    /**
     * @brief Assigns a value to every element of the Vector.
     *
     * @param rhs Value assigned to each element.
     */
    void operator=(const ValueType& rhs);

    /**
     * @brief Assigns the contents of another Vector.
     *
     * @param rhs Vector to copy from.
     *
     * @warning The size of this Vector is adjusted to match @p rhs if necessary.
     */
    void operator=(const Vector<ValueType>& rhs);

    /**
     * @brief Move-assignment operator — transfers ownership of the data buffer.
     *
     * Swaps the data pointer and size from rhs in O(1) without launching any GPU
     * kernel.  The old buffer owned by *this is freed first (after a fence if on GPU).
     * After the move, rhs is left in an empty (size=0, data=nullptr) state.
     * The executor is unchanged — exec_ is const and cannot be moved.
     *
     * @warning Invalidates any existing View objects that point into *this.
     * * @param rhs Vector whose resources are transferred to this Vector.
     * @return A reference to this Vector.
     */
    Vector<ValueType>& operator=(Vector<ValueType>&& rhs) noexcept;

    /**
     * @brief Adds another Vector element-wise.
     *
     * @param rhs Vector to add to this Vector.
     * @return This Vector after the element-wise addition.
     */
    Vector<ValueType>& operator+=(const Vector<ValueType>& rhs);

    /**
     * @brief Subtracts another Vector element-wise.
     *
     * @param rhs Vector to subtract from this Vector.
     * @return A reference to this Vector after the element-wise subtraction.
     */
    Vector<ValueType>& operator-=(const Vector<ValueType>& rhs);

    /**
     * @brief Multiplies two Vectors element-wise.
     *
     * @param rhs Vector to multiply with this Vector.
     * @return A Vector containing the element-wise product.
     *
     * @note This operation is available only for ValueType combinations for which
     *       element-wise multiplication is well-defined. Types with ambiguous
     *       multiplication semantics, such as vec3, are excluded.
     *       See https://eel.is/c++draft/expr.prim.req for information on
     *       C++ requires-expressions.
     */
    [[nodiscard]] Vector<ValueType> operator*(const Vector<ValueType>& rhs)
        requires requires(ValueType a, ValueType b) { a* b; };

    /**
     * @brief Multiplies every element of the Vector by a scalar.
     *
     * @param rhs Scalar multiplier.
     * @return A Vector containing the scaled values.
     *
     * @note This operation is available only for ValueType combinations for which
     *       multiplication by scalar is well-defined. Types with ambiguous
     *       multiplication semantics, such as vec3, are excluded.
     *       See https://eel.is/c++draft/expr.prim.req for information on
     *       C++ requires-expressions.
     */
    [[nodiscard]] Vector<ValueType> operator*(const scalar rhs)
        requires requires(ValueType a, scalar b) { a* b; };

    /**
     * @brief Multiplies this Vector element-wise by another Vector.
     *
     * @param rhs Vector to multiply with this Vector.
     * @return A reference to this Vector after the element-wise multiplication.
     *
     * @note This operation is available only for ValueType combinations for which
     *       element-wise multiplication is well-defined. Types with ambiguous
     *       multiplication semantics, such as vec3, are excluded.
     *       See https://eel.is/c++draft/expr.prim.req for information on
     *       C++ requires-expressions.
     */
    Vector<ValueType>& operator*=(const Vector<ValueType>& rhs)
        requires requires(ValueType a, ValueType b) { a *= b; };

    /**
     * @brief Multiplies every element of the Vector by a scalar.
     *
     * @param rhs Scalar multiplier.
     * @return A reference to this Vector after the scalar multiplication.
     *
     * @note This operation is available only for ValueType combinations for which
     *       multiplication by scalar is well-defined. Types with ambiguous
     *       multiplication semantics, such as vec3, are excluded.
     *       See https://eel.is/c++draft/expr.prim.req for information on
     *       C++ requires-expressions.
     */
    Vector<ValueType>& operator*=(const scalar rhs)
        requires requires(ValueType a, scalar b) { a *= b; };

    /**
     * @brief Resizes the Vector.
     *
     * @param size New number of elements.
     */
    void resize(const localIdx size);
    void resize(const localIdx size);

    /**
     * @brief Returns a pointer to the underlying Vector data.
     *
     * @return Pointer to the first element of the Vector.
     */
    [[nodiscard]] ValueType* data() { return data_; }

    /**
     * @brief Returns a pointer to the underlying Vector data.
     *
     * @return Const pointer to the first element of the Vector.
     */
    [[nodiscard]] const ValueType* data() const { return data_; }

    /**
     * @brief Returns a pointer to the first element of the Vector.
     *
     * @return Pointer to the first element.
     */
    [[nodiscard]] ValueType* begin() { return data_; }

    /**
     * @brief Returns a const pointer to the first element of the Vector.
     *
     * @return Const pointer to the first element.
     */
    [[nodiscard]] const ValueType* begin() const { return data_; }

    /**
     * @brief Returns a pointer one past the last element of the Vector.
     *
     * @return Pointer one past the last element.
     */
    [[nodiscard]] ValueType* end() { return data_ + size(); }

    /**
     * @brief Returns a const pointer one past the last element of the Vector.
     *
     * @return Const pointer one past the last element.
     */
    [[nodiscard]] const ValueType* end() const { return data_ + size(); }

    /**
     * @brief Returns the executor associated with the Vector.
     *
     * @return Reference to the Vector's executor.
     */
    [[nodiscard]] const Executor& exec() const { return exec_; }

    /**
     * @brief Returns the number of elements in the Vector.
     *
     * @return Number of elements.
     */
    [[nodiscard]] localIdx size() const { return size_; }

    /**
     * @brief Returns the number of elements as a label.
     *
     * @return Number of elements converted to label.
     */
    [[nodiscard]] label ssize() const { return static_cast<label>(size_); }

    /**
     * @brief Checks whether the Vector contains no elements.
     *
     * @return True if the Vector is empty, otherwise false.
     */
    [[nodiscard]] bool empty() const { return size() == 0; }

    /**
     * @brief Prevents creating a View from a temporary Vector.
     *
     * A View does not own the referenced data. Allowing a View to be created from
     * a temporary Vector could therefore result in a dangling view.
     */
    View<ValueType> view() && = delete;

    View<const ValueType> view() const&& = delete;

    /**
     * @brief Returns a non-owning view of the Vector data.
     *
     * @return View of the Vector data.
     */
    [[nodiscard]] View<ValueType> view() &
    {
        return View<ValueType>(data_, static_cast<size_t>(size_));
    }

    /**
     * @brief Returns a read-only, non-owning view of the Vector data.
     *
     * @return Read-only view of the Vector data.
     */
    [[nodiscard]] View<const ValueType> view() const&
    {
        return View<const ValueType>(data_, static_cast<size_t>(size_));
    }

    /**
     * @brief Prevents creating a view of a temporary Vector.
     *
     * A View does not own the referenced data, so a view of a temporary Vector
     * could become dangling.
     */
    [[nodiscard]] View<ValueType> view(std::pair<localIdx, localIdx> range) && = delete;

    [[nodiscard]] View<const ValueType> view(std::pair<localIdx, localIdx> range) const&& = delete;

    /**
     * @brief Returns a non-owning view of a range of the Vector.
     *
     * @param range Half-open index range [first, last).
     * @return View of the specified range.
     */
    [[nodiscard]] View<ValueType> view(std::pair<localIdx, localIdx> range) &
    {
        return View<ValueType>(
            data_ + range.first, static_cast<size_t>(range.second - range.first)
        );
    }

    /**
     * @brief Returns a read-only, non-owning view of a range of the Vector.
     *
     * @param range Half-open index range [first, last).
     * @return Read-only view of the specified range.
     */
    [[nodiscard]] View<const ValueType> view(std::pair<localIdx, localIdx> range) const&
    {
        return View<const ValueType>(
            data_ + range.first, static_cast<size_t>(range.second - range.first)
        );
    }

    /**
     * @brief Returns the index range of the Vector.
     *
     * @return The half-open index range [0, size()).
     */
    [[nodiscard]] std::pair<localIdx, localIdx> range() const { return {0, size()}; }

private:

    localIdx size_ {0};         //!< Size of the field.
    ValueType* data_ {nullptr}; //!< Pointer to the field data.
    const Executor exec_;       //!< Executor associated with the field. (CPU, GPU, openMP, etc.)

    /**
     * @brief Checks if two Vectors have the same size and the same executor.
     * @param rhs Vector to validate against this Vector.
     */
    void validateOtherVector(const Vector<ValueType>& rhs) const;
};

/**
 * @brief Adds two Vectors element-wise.
 *
 * @param lhs Left-hand Vector.
 * @param rhs Right-hand Vector.
 * @return A Vector containing the element-wise sum.
 */
template<typename ValueType>
[[nodiscard]] Vector<ValueType> operator+(Vector<ValueType> lhs, const Vector<ValueType>& rhs);

/**
 * @brief Subtracts two Vectors element-wise.
 *
 * @param lhs Left-hand Vector.
 * @param rhs Right-hand Vector.
 * @return A Vector containing the element-wise difference.
 */
template<typename ValueType>
[[nodiscard]] Vector<ValueType> operator-(Vector<ValueType> lhs, const Vector<ValueType>& rhs);

} // namespace NeoN
