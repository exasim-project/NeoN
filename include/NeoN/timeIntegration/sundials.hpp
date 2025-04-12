// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include <concepts>
#include <functional>
#include <memory>

#include <sundials/sundials_nvector.h>
#include <sundials/sundials_core.hpp>
#include <nvector/nvector_serial.h>
#include <nvector/nvector_kokkos.hpp>
#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_erkstep.h>

#include "NeoN/core/error.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/fields/field.hpp"

namespace NeoN::sundials
{

/**
 * @brief Custom deleter for SUNContext shared pointers.
 * @param ctx Pointer to the SUNContext to be freed, can be nullptr.
 * @details Safely frees the context if it's the last reference.
 */
inline auto SUN_CONTEXT_DELETER = [](SUNContext* ctx)
{
    if (ctx != nullptr)
    {
        SUNContext_Free(ctx);
    }
};

/**
 * @brief Custom deleter for explicit type RK solvers (ERK, ARK, etc) for the unique pointers.
 * @param ark Pointer to the ark memory to be freed, can be nullptr.
 * @details Safely frees the ark memory.
 */
inline auto SUN_ARK_DELETER = [](char* ark)
{
    if (ark != nullptr)
    {
        void* arkodMem = reinterpret_cast<void*>(ark);
        ARKodeFree(&arkodMem);
    }
};

/**
 * @brief Maps dictionary keywords to SUNDIALS RKButcher tableau identifiers.
 * @param key The name of the explicit Runge-Kutta method.
 * @return ARKODE_ERKTableID for the corresponding Butcher tableau.
 * @throws Runtime error for unsupported methods.
 */
inline ARKODE_ERKTableID stringToERKTable(const std::string& key)
{
    if (key == "Forward-Euler") return ARKODE_FORWARD_EULER_1_1;
    if (key == "Heun")
    {
        NF_ERROR_EXIT("Currently unsupported until field time step-stage indexing resolved.");
        return ARKODE_HEUN_EULER_2_1_2;
    }
    if (key == "Midpoint")
    {
        NF_ERROR_EXIT("Currently unsupported until field time step-stage indexing resolved.");
        return ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2;
    }
    NF_ERROR_EXIT(
        "Unsupported Runge-Kutta time integration method selectied: " + key + ".\n"
        + "Supported methods are: Forward-Euler, Heun, Midpoint."
    );
    return ARKODE_ERK_NONE; // avoids compiler warnings.
}

/**
 * @brief Converts NeoN Field data to SUNDIALS N_Vec3 format.
 * @tparam SKVec3Type The SUNDIALS Kokkos vector type
 * @tparam ValueType The field data type
 * @param field Source NeoN field
 * @param vector Target SUNDIALS N_Vec3
 * @warning Assumes matching initialization and size between field and vector
 */
template<typename SKVec3Type, typename ValueType>
void fieldToSunNVec3Impl(const NeoN::Field<ValueType>& field, N_Vec3& vector)
{
    auto view = ::sundials::kokkos::GetVec<SKVec3Type>(vector)->View();
    auto fieldView = field.view();
    NeoN::parallelFor(
        field.exec(), field.range(), KOKKOS_LAMBDA(const size_t i) { view(i) = fieldView[i]; }
    );
};

/**
 * @brief Dispatcher for field to N_Vec3 conversion based on executor type.
 * @tparam ValueType The field data type
 * @param field Source NeoN field
 * @param vector Target SUNDIALS N_Vec3
 * @throws Runtime error for unsupported executors
 */
template<typename ValueType>
void fieldToSunNVec3(const NeoN::Field<ValueType>& field, N_Vec3& vector)
{
    // CHECK FOR N_Vec3 on correct space in DEBUG
    if (std::holds_alternative<NeoN::GPUExecutor>(field.exec()))
    {
        fieldToSunNVec3Impl<::sundials::kokkos::Vec3<Kokkos::DefaultExecutionSpace>>(field, vector);
        return;
    }
    if (std::holds_alternative<NeoN::CPUExecutor>(field.exec()))
    {
        fieldToSunNVec3Impl<::sundials::kokkos::Vec3<Kokkos::DefaultHostExecutionSpace>>(
            field, vector
        );
        return;
    }
    if (std::holds_alternative<NeoN::SerialExecutor>(field.exec()))
    {
        fieldToSunNVec3Impl<::sundials::kokkos::Vec3<Kokkos::Serial>>(field, vector);
        return;
    }
    NF_ERROR_EXIT("Unsupported NeoN executor for field.");
};

/**
 * @brief Converts SUNDIALS N_Vec3 data back to NeoN Field format.
 * @tparam SKVec3Type The SUNDIALS Kokkos vector type
 * @tparam ValueType The field data type
 * @param vector Source SUNDIALS N_Vec3
 * @param field Target NeoN field
 * @warning Assumes matching initialization and size between vector and field
 */
template<typename SKVec3Type, typename ValueType>
void sunNVec3ToFieldImpl(const N_Vec3& vector, NeoN::Field<ValueType>& field)
{
    auto view = ::sundials::kokkos::GetVec<SKVec3Type>(vector)->View();
    ValueType* fieldData = field.data();
    NeoN::parallelFor(
        field.exec(), field.range(), KOKKOS_LAMBDA(const size_t i) { fieldData[i] = view(i); }
    );
};

/**
 * @brief Dispatcher for N_Vec3 to field conversion based on executor type.
 * @tparam ValueType The field data type
 * @param vector Source SUNDIALS N_Vec3
 * @param field Target NeoN field
 */
template<typename ValueType>
void sunNVec3ToField(const N_Vec3& vector, NeoN::Field<ValueType>& field)
{
    if (std::holds_alternative<NeoN::GPUExecutor>(field.exec()))
    {
        sunNVec3ToFieldImpl<::sundials::kokkos::Vec3<Kokkos::DefaultExecutionSpace>>(vector, field);
        return;
    }
    if (std::holds_alternative<NeoN::CPUExecutor>(field.exec()))
    {
        sunNVec3ToFieldImpl<::sundials::kokkos::Vec3<Kokkos::DefaultHostExecutionSpace>>(
            vector, field
        );
        return;
    }
    if (std::holds_alternative<NeoN::SerialExecutor>(field.exec()))
    {
        sunNVec3ToFieldImpl<::sundials::kokkos::Vec3<Kokkos::Serial>>(vector, field);
        return;
    }
    NF_ERROR_EXIT("Unsupported NeoN executor for field.");
};

/**
 * @brief Performs a single explicit Runge-Kutta stage evaluation.
 * @param t Current time value
 * @param y Current solution vector
 * @param ydot Output RHS vector
 * @param userData Pointer to Expression object
 * @return 0 on success, non-zero on error
 *
 * @details This is our implementation of the RHS of explicit spacial integration, to be integrated
 * in time. In our case user_data is a unique_ptr to an expression. In this function a 'working
 * source' vector is created and parsed to the explicitOperation, which should contain the field
 * variable at the start of the time step. Currently 'multi-stage RK' is not supported until y
 * can be copied to this field.
 */
template<typename SolutionFieldType>
int explicitRKSolve([[maybe_unused]] sunrealtype t, N_Vec3 y, N_Vec3 ydot, void* userData)
{
    // Pointer wrangling
    using ValueType = typename SolutionFieldType::FieldValueType;
    NeoN::dsl::Expression<ValueType>* pdeExpre =
        reinterpret_cast<NeoN::dsl::Expression<ValueType>*>(userData);
    sunrealtype* yDotArray = N_VGetArrayPointer(ydot);
    sunrealtype* yArray = N_VGetArrayPointer(y);

    NF_ASSERT(
        yDotArray != nullptr && yArray != nullptr && pdeExpre != nullptr,
        "Failed to dereference pointers in sundails."
    );

    size_t size = static_cast<size_t>(N_VGetLength(y));
    // Copy initial value from y to source.
    NeoN::Field<NeoN::scalar> source = pdeExpre->explicitOperation(size) * -1.0; // compute spatial
    if (std::holds_alternative<NeoN::GPUExecutor>(pdeExpre->exec()))
    {
        Kokkos::fence();
    }
    NeoN::sundials::fieldToSunNVec3(source, ydot); // assign rhs to ydot.
    return 0;
}

namespace detail
{

/**
 * @brief Initializes a vector wrapper with specified size and context.
 * @tparam Vec3 Vec3 wrapper type implementing initNVec3 interface
 * @param[in] size Number of elements
 * @param[in] context SUNDIALS context for vector operations
 * @param[in,out] vec Vec3 to initialize
 */
template<typename Vec3>
void initNVec3(size_t size, std::shared_ptr<SUNContext> context, Vec3& vec)
{
    vec.initNVec3(size, context);
}

/**
 * @brief Provides const access to underlying N_Vec3.
 * @tparam Vec3 Vec3 wrapper type implementing NVec3 interface
 * @param vec Source vector wrapper
 * @return Const reference to wrapped N_Vec3
 */
template<typename Vec3>
const N_Vec3& sunNVec3(const Vec3& vec)
{
    return vec.sunNVec3();
}

/**
 * @brief Provides mutable access to underlying N_Vec3.
 * @tparam Vec3 Vec3 wrapper type implementing NVec3 interface
 * @param[in,out] vec Source vector wrapper
 * @return Mutable reference to wrapped N_Vec3
 */
template<typename Vec3>
N_Vec3& sunNVec3(Vec3& vec)
{
    return vec.sunNVec3();
}
}

/**
 * @brief Serial executor SUNDIALS Kokkos vector wrapper.
 * @tparam ValueType The vector data type
 * @details Provides RAII management of SUNDIALS Kokkos vectors for serial execution.
 */
template<typename ValueType>
class SKVec3Serial
{
public:

    SKVec3Serial() {};
    ~SKVec3Serial() = default;
    SKVec3Serial(const SKVec3Serial& other) : kvector_(other.kvector_), svector_(other.kvector_) {};
    SKVec3Serial(SKVec3Serial&& other) noexcept
        : kvector_(std::move(other.kvector_)), svector_(std::move(other.svector_)) {};
    SKVec3Serial& operator=(const SKVec3Serial& other) = delete;
    SKVec3Serial& operator=(SKVec3Serial&& other) = delete;


    using KVec3 = ::sundials::kokkos::Vec3<Kokkos::Serial>;
    void initNVec3(size_t size, std::shared_ptr<SUNContext> context)
    {
        kvector_ = KVec3(size, *context);
        svector_ = kvector_;
    };
    const N_Vec3& sunNVec3() const { return svector_; };
    N_Vec3& sunNVec3() { return svector_; };

private:

    KVec3 kvector_ {}; /**< The Sundails, kokkos initial conditions vector (do not use).*/
    N_Vec3 svector_ {nullptr};
};

/**
 * @brief Host default executor SUNDIALS Kokkos vector wrapper.
 * @tparam ValueType The vector data type
 * @details Provides RAII management of SUNDIALS Kokkos vectors for CPU execution.
 */
template<typename ValueType>
class SKVec3HostDefault
{
public:

    using KVec3 = ::sundials::kokkos::Vec3<Kokkos::DefaultHostExecutionSpace>;

    SKVec3HostDefault() = default;
    ~SKVec3HostDefault() = default;
    SKVec3HostDefault(const SKVec3HostDefault& other)
        : kvector_(other.kvector_), svector_(other.kvector_) {};
    SKVec3HostDefault(SKVec3HostDefault&& other) noexcept
        : kvector_(std::move(other.kvector_)), svector_(std::move(other.svector_)) {};
    SKVec3HostDefault& operator=(const SKVec3HostDefault& other) = delete;
    SKVec3HostDefault& operator=(SKVec3HostDefault&& other) = delete;

    void initNVec3(size_t size, std::shared_ptr<SUNContext> context)
    {
        kvector_ = KVec3(size, *context);
        svector_ = kvector_;
    };
    const N_Vec3& sunNVec3() const { return svector_; };
    N_Vec3& sunNVec3() { return svector_; };

private:

    KVec3 kvector_ {}; /**< The Sundails, kokkos initial conditions vector (do not use).*/
    N_Vec3 svector_ {nullptr};
};

/**
 * @brief Default executor SUNDIALS Kokkos vector wrapper.
 * @tparam ValueType The vector data type
 * @details Provides RAII management of SUNDIALS Kokkos vectors for GPU execution.
 */
template<typename ValueType>
class SKVec3Default
{
public:

    using KVec3 = ::sundials::kokkos::Vec3<Kokkos::DefaultExecutionSpace>;

    SKVec3Default() = default;
    ~SKVec3Default() = default;
    SKVec3Default(const SKVec3Default& other)
        : kvector_(other.kvector_), svector_(other.kvector_) {};
    SKVec3Default(SKVec3Default&& other) noexcept
        : kvector_(std::move(other.kvector_)), svector_(std::move(other.svector_)) {};
    SKVec3Default& operator=(const SKVec3Default& other) = delete;
    SKVec3Default& operator=(SKVec3Default&& other) = delete;

    void initNVec3(size_t size, std::shared_ptr<SUNContext> context)
    {
        kvector_ = KVec3(size, *context);
        svector_ = kvector_;
    };

    const N_Vec3& sunNVec3() const { return svector_; };

    N_Vec3& sunNVec3() { return svector_; };

private:

    KVec3 kvector_ {}; /**< The Sundails, kokkos initial conditions vector (do not use).*/
    N_Vec3 svector_ {nullptr};
};

/**
 * @brief Unified interface for SUNDIALS Kokkos vector management.
 * @tparam ValueType The vector data type
 * @details Manages executor-specific vector implementations through variant storage.
 * Provides common interface for vector initialization and access.
 */
template<typename ValueType>
class SKVec3
{
public:

    using SKVec3SerialV = SKVec3Serial<ValueType>;
    using SKVec3HostDefaultV = SKVec3HostDefault<ValueType>;
    using SKDefaultVec3V = SKVec3Default<ValueType>;
    using SKVec3Variant = std::variant<SKVec3SerialV, SKVec3HostDefaultV, SKDefaultVec3V>;

    /**
     * @brief Default constructor. Initializes with host-default vector.
     */
    SKVec3() { vector_.template emplace<SKVec3HostDefaultV>(); };

    /**
     * @brief Default destructor.
     */
    ~SKVec3() = default;

    /**
     * @brief Copy constructor.
     * @param[in] other Source SKVec3 to copy from
     */
    SKVec3(const SKVec3&) = default;

    /**
     * @brief Copy assignment operator (deleted).
     */
    SKVec3& operator=(const SKVec3&) = delete;

    /**
     * @brief Move constructor.
     * @param[in] other Source SKVec3 to move from
     */
    SKVec3(SKVec3&&) noexcept = default;

    /**
     * @brief Move assignment operator (deleted).
     */
    SKVec3& operator=(SKVec3&&) noexcept = delete;

    /**
     * @brief Sets appropriate vector implementation based on executor type.
     * @param[in] exec NeoN executor specifying computation space
     */
    void setExecutor(const NeoN::Executor& exec)
    {
        if (std::holds_alternative<NeoN::GPUExecutor>(exec))
        {
            vector_.template emplace<SKDefaultVec3V>();
            return;
        }
        if (std::holds_alternative<NeoN::CPUExecutor>(exec))
        {
            vector_.template emplace<SKVec3HostDefaultV>();
            return;
        }
        if (std::holds_alternative<NeoN::SerialExecutor>(exec))
        {
            vector_.template emplace<SKVec3SerialV>();
            return;
        }

        NF_ERROR_EXIT(
            "Unsupported NeoN executor: "
            << std::visit([](const auto& e) { return e.name(); }, exec) << "."
        );
    }

    /**
     * @brief Initializes underlying vector with given size and context.
     * @param size Number of vector elements
     * @param context SUNDIALS context for vector operations
     */
    void initNVec3(size_t size, std::shared_ptr<SUNContext> context)
    {
        std::visit([size, &context](auto& vec) { detail::initNVec3(size, context, vec); }, vector_);
    }

    /**
     * @brief Gets const reference to underlying N_Vec3.
     * @return Const reference to wrapped SUNDIALS N_Vec3
     */
    const N_Vec3& sunNVec3() const
    {
        return std::visit(
            [](const auto& vec) -> const N_Vec3& { return detail::sunNVec3(vec); }, vector_
        );
    }

    /**
     * @brief Gets mutable reference to underlying N_Vec3.
     * @return Mutable reference to wrapped SUNDIALS N_Vec3
     */
    N_Vec3& sunNVec3()
    {
        return std::visit([](auto& vec) -> N_Vec3& { return detail::sunNVec3(vec); }, vector_);
    }

    /**
     * @brief Gets const reference to variant storing implementation.
     * @return Const reference to vector variant
     */
    const SKVec3Variant& variant() const { return vector_; }

    /**
     * @brief Gets mutable reference to variant storing implementation.
     * @return Mutable reference to vector variant
     */
    SKVec3Variant& variant() { return vector_; }

private:

    SKVec3Variant vector_; /**< Variant storing executor-specific vector implementation */
};
}
