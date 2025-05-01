// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/vector/vector.hpp"

namespace NeoN
{

template<typename ValueType>
Vector<ValueType>::Vector(const Executor& exec, localIdx size)
    : size_(size), data_(nullptr), exec_(exec)
{
    void* ptr = nullptr;
    std::visit(
        [&ptr, size](const auto& concreteExec)
        { ptr = concreteExec.alloc(static_cast<size_t>(size) * sizeof(ValueType)); },
        exec_
    );
    data_ = static_cast<ValueType*>(ptr);
}

template<typename ValueType>
Vector<ValueType>::Vector(
    const Executor& exec, const ValueType* in, localIdx size, Executor hostExec
)
    : size_(size), data_(nullptr), exec_(exec)
{
    void* ptr = nullptr;
    std::visit(
        [&ptr, size](const auto& concreteExec)
        { ptr = concreteExec.alloc(static_cast<size_t>(size) * sizeof(ValueType)); },
        exec_
    );
    data_ = static_cast<ValueType*>(ptr);
    std::visit(detail::deepCopyVisitor<ValueType>(size_, in, data_), hostExec, exec_);
}

template<typename ValueType>
Vector<ValueType>::Vector(const Executor& exec, localIdx size, ValueType value)
    : size_(size), data_(nullptr), exec_(exec)
{
    void* ptr = nullptr;
    std::visit(
        [&ptr, size](const auto& execu)
        { ptr = execu.alloc(static_cast<size_t>(size) * sizeof(ValueType)); },
        exec_
    );
    data_ = static_cast<ValueType*>(ptr);
    NeoN::fill(*this, value);
}

template<typename ValueType>
Vector<ValueType>::Vector(const Executor& exec, std::vector<ValueType> in)
    : Vector(exec, in.data(), static_cast<localIdx>(in.size()))
{}

template<typename ValueType>
Vector<ValueType>::Vector(const Executor& exec, const Vector<ValueType>& in)
    : Vector(exec, in.data(), in.size(), in.exec())
{}

template<typename ValueType>
Vector<ValueType>::Vector(const Vector<ValueType>& rhs)
    : Vector(rhs.exec(), rhs.data(), rhs.size(), rhs.exec())
{}

template<typename ValueType>
Vector<ValueType>::Vector(Vector<ValueType>&& rhs) noexcept
    : size_(rhs.size_), data_(rhs.data_), exec_(rhs.exec_)
{
    rhs.data_ = nullptr;
    rhs.size_ = 0;
}

template<typename ValueType>
Vector<ValueType>::~Vector()
{
    std::visit([this](const auto& exec) { exec.free(data_); }, exec_);
    data_ = nullptr;
}

template<typename ValueType>
[[nodiscard]] Vector<ValueType> Vector<ValueType>::copyToExecutor(Executor dstExec) const
{
    if (dstExec == exec_) return *this;

    Vector<ValueType> result(dstExec, size_);
    std::visit(detail::deepCopyVisitor(size_, data_, result.data()), exec_, dstExec);

    return result;
}

template<typename ValueType>
[[nodiscard]] Vector<ValueType> Vector<ValueType>::copyToHost() const
{
    return copyToExecutor(SerialExecutor());
}

template<typename ValueType>
void Vector<ValueType>::copyToHost(Vector<ValueType>& result)
{
    NF_DEBUG_ASSERT(
        result.size() == size_, "Parsed Vector size not the same as current field size"
    );
    result = copyToExecutor(SerialExecutor());
}

template<typename ValueType>
void Vector<ValueType>::operator=(const ValueType& rhs)
{
    fill(*this, rhs);
}

template<typename ValueType>
void Vector<ValueType>::operator=(const Vector<ValueType>& rhs)
{
    NF_ASSERT(exec_ == rhs.exec_, "Executors are not the same");
    if (this->size() != rhs.size())
    {
        this->resize(rhs.size());
    }
    setContainer(*this, rhs.view());
}

template<typename ValueType>
Vector<ValueType>& Vector<ValueType>::operator+=(const Vector<ValueType>& rhs)
{
    validateOtherVector(rhs);
    add(*this, rhs);
    return *this;
}

template<typename ValueType>
Vector<ValueType>& Vector<ValueType>::operator-=(const Vector<ValueType>& rhs)
{
    validateOtherVector(rhs);
    sub(*this, rhs);
    return *this;
}

template<typename ValueType>
Vector<ValueType> Vector<ValueType>::operator*(const Vector<ValueType>& rhs)
    requires requires(ValueType a, ValueType b) { a* b; }
{
    validateOtherVector(rhs);
    Vector<ValueType> result(exec_, size_);
    result = *this;
    mul(result, rhs);
    return result;
}

template<typename ValueType>
Vector<ValueType> Vector<ValueType>::operator*(const ValueType rhs)
    requires requires(ValueType a, ValueType b) { a* b; }
{
    Vector<ValueType> result(exec_, size_);
    result = *this;
    scalarMul(result, rhs);
    return result;
}

template<typename ValueType>
Vector<ValueType>& Vector<ValueType>::operator*=(const Vector<ValueType>& rhs)
    requires requires(ValueType a, ValueType b) { a *= b; }
{
    validateOtherVector(rhs);
    Vector<ValueType>& result = *this;
    mul(result, rhs);
    return result;
}

template<typename ValueType>
Vector<ValueType>& Vector<ValueType>::operator*=(const ValueType rhs)
    requires requires(ValueType a, ValueType b) { a *= b; }
{
    Vector<ValueType>& result = *this;
    scalarMul(result, rhs);
    return result;
}

template<typename ValueType>
void Vector<ValueType>::resize(const localIdx size)
{
    void* ptr = nullptr;
    if (!empty())
    {
        std::visit(
            [this, &ptr, size](const auto& exec)
            { ptr = exec.realloc(this->data_, static_cast<size_t>(size) * sizeof(ValueType)); },
            exec_
        );
    }
    else
    {
        std::visit(
            [&ptr, size](const auto& exec)
            { ptr = exec.alloc(static_cast<size_t>(size) * sizeof(ValueType)); },
            exec_
        );
    }
    data_ = static_cast<ValueType*>(ptr);
    size_ = size;
}

template<typename ValueType>
void Vector<ValueType>::validateOtherVector(const Vector<ValueType>& rhs) const
{
    NF_DEBUG_ASSERT(size() == rhs.size(), "Vectors are not the same size.");
    NF_DEBUG_ASSERT(exec() == rhs.exec(), "Executors are not the same.");
}

template<typename ValueType>
[[nodiscard]] Vector<ValueType> operator+(Vector<ValueType> lhs, const Vector<ValueType>& rhs)
{
    lhs += rhs;
    return lhs;
}

template<typename ValueType>
[[nodiscard]] Vector<ValueType> operator-(Vector<ValueType> lhs, const Vector<ValueType>& rhs)
{
    lhs -= rhs;
    return lhs;
}

// template class instantiation
template class Vector<uint32_t>;
template class Vector<uint64_t>;
template class Vector<int32_t>;
template class Vector<int64_t>;
template class Vector<float>;
template class Vector<double>;
template class Vector<Vec3>;

// operator instantiation
#define OPERATOR_INSTANTIATION(Type)                                                               \
    /* free function opperators with additional requirements  */                                   \
    template Vector<Type> operator+(Vector<Type> lhs, const Vector<Type>& rhs);                    \
    template Vector<Type> operator-(Vector<Type> lhs, const Vector<Type>& rhs);

OPERATOR_INSTANTIATION(uint32_t);
OPERATOR_INSTANTIATION(uint64_t);
OPERATOR_INSTANTIATION(int32_t);
OPERATOR_INSTANTIATION(int64_t);
OPERATOR_INSTANTIATION(float);
OPERATOR_INSTANTIATION(double);
OPERATOR_INSTANTIATION(Vec3);


} // namespace NeoN
