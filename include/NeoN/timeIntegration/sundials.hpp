// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#if NN_WITH_SUNDIALS

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
 * @brief Converts NeoN Vector data to SUNDIALS N_Vector format.
 * @tparam SKVectorType The SUNDIALS Kokkos vector type
 * @tparam ValueType The field data type
 * @param field Source NeoN field
 * @param vector Target SUNDIALS N_Vector
 * @warning Assumes matching initialization and size between field and vector
 */
template<typename SKVectorType, typename ValueType>
void fieldToSunNVectorImpl(const NeoN::Vector<ValueType>& field, N_Vector& vector)
{
    auto view = ::sundials::kokkos::GetVec<SKVectorType>(vector)->View();
    auto fieldView = field.view();
    NeoN::parallelFor(
        field.exec(),
        field.range(),
        KOKKOS_LAMBDA(const localIdx i) { view(i) = fieldView[i]; },
        "fieldToSunVector"
    );
};

/**
 * @brief Dispatcher for field to N_Vector conversion based on executor type.
 * @tparam ValueType The field data type
 * @param field Source NeoN field
 * @param vector Target SUNDIALS N_Vector
 * @throws Runtime error for unsupported executors
 */
template<typename ValueType>
void fieldToSunNVector(const NeoN::Vector<ValueType>& field, N_Vector& vector)
{
    // CHECK FOR N_Vector on correct space in DEBUG
    if (std::holds_alternative<NeoN::GPUExecutor>(field.exec()))
    {
        fieldToSunNVectorImpl<::sundials::kokkos::Vector<Kokkos::DefaultExecutionSpace>>(
            field, vector
        );
        return;
    }
    if (std::holds_alternative<NeoN::CPUExecutor>(field.exec()))
    {
        fieldToSunNVectorImpl<::sundials::kokkos::Vector<Kokkos::DefaultHostExecutionSpace>>(
            field, vector
        );
        return;
    }
    if (std::holds_alternative<NeoN::SerialExecutor>(field.exec()))
    {
        fieldToSunNVectorImpl<::sundials::kokkos::Vector<Kokkos::Serial>>(field, vector);
        return;
    }
    NF_ERROR_EXIT("Unsupported NeoN executor for field.");
};

/**
 * @brief Converts SUNDIALS N_Vector data back to NeoN Vector format.
 * @tparam SKVectorType The SUNDIALS Kokkos vector type
 * @tparam ValueType The field data type
 * @param vector Source SUNDIALS N_Vector
 * @param field Target NeoN field
 * @warning Assumes matching initialization and size between vector and field
 */
template<typename SKVectorType, typename ValueType>
void sunNVectorToVectorImpl(const N_Vector& vector, NeoN::Vector<ValueType>& field)
{
    auto view = ::sundials::kokkos::GetVec<SKVectorType>(vector)->View();
    ValueType* fieldData = field.data();
    NeoN::parallelFor(
        field.exec(),
        field.range(),
        KOKKOS_LAMBDA(const localIdx i) { fieldData[i] = view(i); },
        "sunNVectorToVectorImpl"
    );
};

/**
 * @brief Dispatcher for N_Vector to field conversion based on executor type.
 * @tparam ValueType The field data type
 * @param vector Source SUNDIALS N_Vector
 * @param field Target NeoN field
 */
template<typename ValueType>
void sunNVectorToVector(const N_Vector& vector, NeoN::Vector<ValueType>& field)
{
    if (std::holds_alternative<NeoN::GPUExecutor>(field.exec()))
    {
        sunNVectorToVectorImpl<::sundials::kokkos::Vector<Kokkos::DefaultExecutionSpace>>(
            vector, field
        );
        return;
    }
    if (std::holds_alternative<NeoN::CPUExecutor>(field.exec()))
    {
        sunNVectorToVectorImpl<::sundials::kokkos::Vector<Kokkos::DefaultHostExecutionSpace>>(
            vector, field
        );
        return;
    }
    if (std::holds_alternative<NeoN::SerialExecutor>(field.exec()))
    {
        sunNVectorToVectorImpl<::sundials::kokkos::Vector<Kokkos::Serial>>(vector, field);
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
template<typename SolutionVectorType>
int explicitRKSolve([[maybe_unused]] sunrealtype t, N_Vector y, N_Vector ydot, void* userData)
{
    // Pointer wrangling
    using ValueType = typename SolutionVectorType::VectorValueType;
    NeoN::dsl::Expression<ValueType>* pdeExpre =
        reinterpret_cast<NeoN::dsl::Expression<ValueType>*>(userData);
    sunrealtype* yDotArray = N_VGetArrayPointer(ydot);
    sunrealtype* yArray = N_VGetArrayPointer(y);

    NF_ASSERT(
        yDotArray != nullptr && yArray != nullptr && pdeExpre != nullptr,
        "Failed to dereference pointers in sundails."
    );

    auto size = static_cast<localIdx>(N_VGetLength(y));
    // Copy initial value from y to source.
    NeoN::Vector<NeoN::scalar> source = pdeExpre->explicitOperation(size) * -1.0; // compute spatial
    fence(pdeExpre->exec());
    NeoN::sundials::fieldToSunNVector(source, ydot); // assign rhs to ydot.
    return 0;
}

namespace detail
{

/**
 * @brief Initializes a vector wrapper with specified size and context.
 * @tparam Vector Vector wrapper type implementing initNVector interface
 * @param[in] size Number of elements
 * @param[in] context SUNDIALS context for vector operations
 * @param[in,out] vec Vector to initialize
 */
template<typename Vector>
void initNVector(size_t size, std::shared_ptr<SUNContext> context, Vector& vec)
{
    vec.initNVector(size, context);
}

/**
 * @brief Provides const access to underlying N_Vector.
 * @tparam Vector Vector wrapper type implementing NVector interface
 * @param vec Source vector wrapper
 * @return Const reference to wrapped N_Vector
 */
template<typename Vector>
const N_Vector& sunNVector(const Vector& vec)
{
    return vec.sunNVector();
}

/**
 * @brief Provides mutable access to underlying N_Vector.
 * @tparam Vector Vector wrapper type implementing NVector interface
 * @param[in,out] vec Source vector wrapper
 * @return Mutable reference to wrapped N_Vector
 */
template<typename Vector>
N_Vector& sunNVector(Vector& vec)
{
    return vec.sunNVector();
}
}

/**
 * @brief Serial executor SUNDIALS Kokkos vector wrapper.
 * @tparam ValueType The vector data type
 * @details Provides RAII management of SUNDIALS Kokkos vectors for serial execution.
 */
template<typename ValueType>
class SKVectorSerial
{
public:

    SKVectorSerial() {};
    ~SKVectorSerial() = default;
    SKVectorSerial(const SKVectorSerial& other)
        : kvector_(other.kvector_), svector_(other.kvector_) {};
    SKVectorSerial(SKVectorSerial&& other) noexcept
        : kvector_(std::move(other.kvector_)), svector_(std::move(other.svector_)) {};
    SKVectorSerial& operator=(const SKVectorSerial& other) = delete;
    SKVectorSerial& operator=(SKVectorSerial&& other) = delete;


    using KVector = ::sundials::kokkos::Vector<Kokkos::Serial>;
    void initNVector(size_t size, std::shared_ptr<SUNContext> context)
    {
        kvector_ = KVector(size, *context);
        svector_ = kvector_;
    };
    const N_Vector& sunNVector() const { return svector_; };
    N_Vector& sunNVector() { return svector_; };

private:

    KVector kvector_ {}; /**< The Sundails, kokkos initial conditions vector (do not use).*/
    N_Vector svector_ {nullptr};
};

/**
 * @brief Host default executor SUNDIALS Kokkos vector wrapper.
 * @tparam ValueType The vector data type
 * @details Provides RAII management of SUNDIALS Kokkos vectors for CPU execution.
 */
template<typename ValueType>
class SKVectorHostDefault
{
public:

    using KVector = ::sundials::kokkos::Vector<Kokkos::DefaultHostExecutionSpace>;

    SKVectorHostDefault() = default;
    ~SKVectorHostDefault() = default;
    SKVectorHostDefault(const SKVectorHostDefault& other)
        : kvector_(other.kvector_), svector_(other.kvector_) {};
    SKVectorHostDefault(SKVectorHostDefault&& other) noexcept
        : kvector_(std::move(other.kvector_)), svector_(std::move(other.svector_)) {};
    SKVectorHostDefault& operator=(const SKVectorHostDefault& other) = delete;
    SKVectorHostDefault& operator=(SKVectorHostDefault&& other) = delete;

    void initNVector(size_t size, std::shared_ptr<SUNContext> context)
    {
        kvector_ = KVector(size, *context);
        svector_ = kvector_;
    };
    const N_Vector& sunNVector() const { return svector_; };
    N_Vector& sunNVector() { return svector_; };

private:

    KVector kvector_ {}; /**< The Sundails, kokkos initial conditions vector (do not use).*/
    N_Vector svector_ {nullptr};
};

/**
 * @brief Default executor SUNDIALS Kokkos vector wrapper.
 * @tparam ValueType The vector data type
 * @details Provides RAII management of SUNDIALS Kokkos vectors for GPU execution.
 */
template<typename ValueType>
class SKVectorDefault
{
public:

    using KVector = ::sundials::kokkos::Vector<Kokkos::DefaultExecutionSpace>;

    SKVectorDefault() = default;
    ~SKVectorDefault() = default;
    SKVectorDefault(const SKVectorDefault& other)
        : kvector_(other.kvector_), svector_(other.kvector_) {};
    SKVectorDefault(SKVectorDefault&& other) noexcept
        : kvector_(std::move(other.kvector_)), svector_(std::move(other.svector_)) {};
    SKVectorDefault& operator=(const SKVectorDefault& other) = delete;
    SKVectorDefault& operator=(SKVectorDefault&& other) = delete;

    void initNVector(size_t size, std::shared_ptr<SUNContext> context)
    {
        kvector_ = KVector(size, *context);
        svector_ = kvector_;
    };

    const N_Vector& sunNVector() const { return svector_; };

    N_Vector& sunNVector() { return svector_; };

private:

    KVector kvector_ {}; /**< The Sundails, kokkos initial conditions vector (do not use).*/
    N_Vector svector_ {nullptr};
};

/**
 * @brief Unified interface for SUNDIALS Kokkos vector management.
 * @tparam ValueType The vector data type
 * @details Manages executor-specific vector implementations through variant storage.
 * Provides common interface for vector initialization and access.
 */
template<typename ValueType>
class SKVector
{
public:

    using SKVectorSerialV = SKVectorSerial<ValueType>;
    using SKVectorHostDefaultV = SKVectorHostDefault<ValueType>;
    using SKDefaultVectorV = SKVectorDefault<ValueType>;
    using SKVectorVariant = std::variant<SKVectorSerialV, SKVectorHostDefaultV, SKDefaultVectorV>;

    /**
     * @brief Default constructor. Initializes with host-default vector.
     */
    SKVector() { vector_.template emplace<SKVectorHostDefaultV>(); };

    /**
     * @brief Default destructor.
     */
    ~SKVector() = default;

    /**
     * @brief Copy constructor.
     * @param[in] other Source SKVector to copy from
     */
    SKVector(const SKVector&) = default;

    /**
     * @brief Copy assignment operator (deleted).
     */
    SKVector& operator=(const SKVector&) = delete;

    /**
     * @brief Move constructor.
     * @param[in] other Source SKVector to move from
     */
    SKVector(SKVector&&) noexcept = default;

    /**
     * @brief Move assignment operator (deleted).
     */
    SKVector& operator=(SKVector&&) noexcept = delete;

    /**
     * @brief Sets appropriate vector implementation based on executor type.
     * @param[in] exec NeoN executor specifying computation space
     */
    void setExecutor(const NeoN::Executor& exec)
    {
        if (std::holds_alternative<NeoN::GPUExecutor>(exec))
        {
            vector_.template emplace<SKDefaultVectorV>();
            return;
        }
        if (std::holds_alternative<NeoN::CPUExecutor>(exec))
        {
            vector_.template emplace<SKVectorHostDefaultV>();
            return;
        }
        if (std::holds_alternative<NeoN::SerialExecutor>(exec))
        {
            vector_.template emplace<SKVectorSerialV>();
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
    void initNVector(size_t size, std::shared_ptr<SUNContext> context)
    {
        std::visit(
            [size, &context](auto& vec) { detail::initNVector(size, context, vec); }, vector_
        );
    }

    /**
     * @brief Gets const reference to underlying N_Vector.
     * @return Const reference to wrapped SUNDIALS N_Vector
     */
    const N_Vector& sunNVector() const
    {
        return std::visit(
            [](const auto& vec) -> const N_Vector& { return detail::sunNVector(vec); }, vector_
        );
    }

    /**
     * @brief Gets mutable reference to underlying N_Vector.
     * @return Mutable reference to wrapped SUNDIALS N_Vector
     */
    N_Vector& sunNVector()
    {
        return std::visit([](auto& vec) -> N_Vector& { return detail::sunNVector(vec); }, vector_);
    }

    /**
     * @brief Gets const reference to variant storing implementation.
     * @return Const reference to vector variant
     */
    const SKVectorVariant& variant() const { return vector_; }

    /**
     * @brief Gets mutable reference to variant storing implementation.
     * @return Mutable reference to vector variant
     */
    SKVectorVariant& variant() { return vector_; }

private:

    SKVectorVariant vector_; /**< Variant storing executor-specific vector implementation */
};
}

#endif
