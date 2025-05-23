// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/timeIntegration/rungeKutta.hpp"

#if NF_WITH_SUNDIALS

namespace NeoN::timeIntegration
{

template<typename SolutionVectorType>
RungeKutta<SolutionVectorType>::RungeKutta(const RungeKutta<SolutionVectorType>& other)
    : Base(other), solution_(other.solution_), initialConditions_(other.initialConditions_),
      pdeExpr_(
          other.pdeExpr_
              ? std::make_unique<NeoN::dsl::Expression<ValueType>>(other.pdeExpr_->exec())
              : nullptr
      )
{
    sunrealtype timeCurrent;
    void* ark = reinterpret_cast<void*>(other.ODEMemory_.get());
    ARKodeGetCurrentTime(ark, &timeCurrent);
    initODEMemory(timeCurrent); // will finalise construction of the ode memory.
}

template<typename SolutionVectorType>
RungeKutta<SolutionVectorType>::RungeKutta(RungeKutta<SolutionVectorType>&& other)
    : Base(std::move(other)), solution_(std::move(other.solution_)),
      initialConditions_(std::move(other.initialConditions_)), context_(std::move(other.context_)),
      ODEMemory_(std::move(other.ODEMemory_)), pdeExpr_(std::move(other.pdeExpr_))
{}

template<typename SolutionVectorType>
void RungeKutta<SolutionVectorType>::solve(
    dsl::Expression<ValueType>& exp, SolutionVectorType& solutionVector, scalar t, const scalar dt
)
{
    // Setup sundials if required, load the current solution for temporal integration
    SolutionVectorType& oldSolutionVector =
        NeoN::finiteVolume::cellCentred::oldTime(solutionVector);
    if (pdeExpr_ == nullptr) initSUNERKSolver(exp, oldSolutionVector, t);
    NeoN::sundials::fieldToSunNVector(oldSolutionVector.internalVector(), solution_.sunNVector());
    void* ark = reinterpret_cast<void*>(ODEMemory_.get());

    // Perform time integration
    ARKodeSetFixedStep(ark, dt);
    NeoN::scalar timeOut;
    auto stepReturn = ARKodeEvolve(ark, t + dt, solution_.sunNVector(), &timeOut, ARK_ONE_STEP);

    // Post step checks
    NF_ASSERT_EQUAL(stepReturn, 0);
    NF_ASSERT_EQUAL(t + dt, timeOut);

    // Copy solution out. (Fence is in sundails free)
    NeoN::sundials::sunNVectorToVector(solution_.sunNVector(), solutionVector.internalVector());
    oldSolutionVector.internalVector() = solutionVector.internalVector();
}

template<typename SolutionVectorType>
std::unique_ptr<TimeIntegratorBase<SolutionVectorType>>
RungeKutta<SolutionVectorType>::clone() const
{
    return std::make_unique<RungeKutta>(*this);
}

template<typename SolutionVectorType>
void RungeKutta<SolutionVectorType>::initSUNERKSolver(
    dsl::Expression<typename SolutionVectorType::VectorValueType>& exp,
    SolutionVectorType& field,
    const scalar t
)
{
    initExpression(exp);
    initSUNContext();
    initSUNVector(field.exec(), static_cast<size_t>(field.internalVector().size()));
    initSUNInitialConditions(field);
    initODEMemory(t);
}

template<typename SolutionVectorType>
void RungeKutta<SolutionVectorType>::initExpression(const dsl::Expression<ValueType>& exp)
{
    pdeExpr_ = std::make_unique<dsl::Expression<ValueType>>(exp);
}

// NOTE: This function triggers an error with the leak checkers/asan
// i dont see it to actually leak memory since we use SUN_CONTEXT_DELETER
// for the time being the we ignore this function by adding it to scripts/san_ignores
// if you figure out whether it actually leaks memory or how to satisfy asan remove this note
// and the function from san_ignores.txt
template<typename SolutionVectorType>
void RungeKutta<SolutionVectorType>::initSUNContext()
{
    if (!context_)
    {
        std::shared_ptr<SUNContext> context(new SUNContext(), sundials::SUN_CONTEXT_DELETER);
        int flag = SUNContext_Create(SUN_COMM_NULL, context.get());
        NF_ASSERT(flag == 0, "SUNContext_Create failed");
        context_.swap(context);
    }
}

template<typename SolutionVectorType>
void RungeKutta<SolutionVectorType>::initSUNVector(const Executor& exec, size_t size)
{
    NF_DEBUG_ASSERT(context_, "SUNContext is a nullptr.");
    solution_.setExecutor(exec);
    solution_.initNVector(size, context_);
    initialConditions_.setExecutor(exec);
    initialConditions_.initNVector(size, context_);
}

template<typename SolutionVectorType>
void RungeKutta<SolutionVectorType>::initSUNInitialConditions(
    const SolutionVectorType& solutionVector
)
{

    NeoN::sundials::fieldToSunNVector(
        solutionVector.internalVector(), initialConditions_.sunNVector()
    );
}

template<typename SolutionVectorType>
void RungeKutta<SolutionVectorType>::initODEMemory(const scalar t)
{
    NF_DEBUG_ASSERT(context_, "SUNContext is a nullptr.");
    NF_DEBUG_ASSERT(pdeExpr_, "PDE expression is a nullptr.");

    void* ark = ERKStepCreate(
        NeoN::sundials::explicitRKSolve<SolutionVectorType>,
        t,
        initialConditions_.sunNVector(),
        *context_
    );
    ODEMemory_.reset(reinterpret_cast<char*>(ark));

    // Initialize ERKStep solver
    ERKStepSetTableNum(
        ark,
        NeoN::sundials::stringToERKTable(
            this->schemeDict_.template get<std::string>("Runge-Kutta-Method")
        )
    );
    ARKodeSetUserData(ark, pdeExpr_.get());
    ARKodeSStolerances(ODEMemory_.get(), 1.0, 1.0); // If we want ARK we will revisit.
}

template class RungeKutta<finiteVolume::cellCentred::VolumeField<scalar>>;
}

#endif
