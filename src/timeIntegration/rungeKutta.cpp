// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/timeIntegration/rungeKutta.hpp"

namespace NeoN::timeIntegration
{

template<typename SolutionFieldType>
RungeKutta<SolutionFieldType>::RungeKutta(const RungeKutta<SolutionFieldType>& other)
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

template<typename SolutionFieldType>
RungeKutta<SolutionFieldType>::RungeKutta(RungeKutta<SolutionFieldType>&& other)
    : Base(std::move(other)), solution_(std::move(other.solution_)),
      initialConditions_(std::move(other.initialConditions_)), context_(std::move(other.context_)),
      ODEMemory_(std::move(other.ODEMemory_)), pdeExpr_(std::move(other.pdeExpr_))
{}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::solve(
    dsl::Expression<ValueType>& exp, SolutionFieldType& solutionField, scalar t, const scalar dt
)
{
    // Setup sundials if required, load the current solution for temporal integration
    SolutionFieldType& oldSolutionField = NeoN::finiteVolume::cellCentred::oldTime(solutionField);
    if (pdeExpr_ == nullptr) initSUNERKSolver(exp, oldSolutionField, t);
    NeoN::sundials::fieldToSunNVector(oldSolutionField.internalField(), solution_.sunNVector());
    void* ark = reinterpret_cast<void*>(ODEMemory_.get());

    // Perform time integration
    ARKodeSetFixedStep(ark, dt);
    NeoN::scalar timeOut;
    auto stepReturn = ARKodeEvolve(ark, t + dt, solution_.sunNVector(), &timeOut, ARK_ONE_STEP);

    // Post step checks
    NF_ASSERT_EQUAL(stepReturn, 0);
    NF_ASSERT_EQUAL(t + dt, timeOut);

    // Copy solution out. (Fence is in sundails free)
    NeoN::sundials::sunNVectorToField(solution_.sunNVector(), solutionField.internalField());
    oldSolutionField.internalField() = solutionField.internalField();
}

template<typename SolutionFieldType>
std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> RungeKutta<SolutionFieldType>::clone() const
{
    return std::make_unique<RungeKutta>(*this);
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initSUNERKSolver(
    dsl::Expression<typename SolutionFieldType::FieldValueType>& exp,
    SolutionFieldType& field,
    const scalar t
)
{
    initExpression(exp);
    initSUNContext();
    initSUNVector(field.exec(), field.internalField().size());
    initSUNInitialConditions(field);
    initODEMemory(t);
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initExpression(const dsl::Expression<ValueType>& exp)
{
    pdeExpr_ = std::make_unique<dsl::Expression<ValueType>>(exp);
}

// NOTE: This function triggers an error with the leak checkers/asan
// i dont see it to actually leak memory since we use SUN_CONTEXT_DELETER
// for the time being the we ignore this function by adding it to scripts/san_ignores
// if you figure out whether it actually leaks memory or how to satisfy asan remove this note
// and the function from san_ignores.txt
template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initSUNContext()
{
    if (!context_)
    {
        std::shared_ptr<SUNContext> context(new SUNContext(), sundials::SUN_CONTEXT_DELETER);
        int flag = SUNContext_Create(SUN_COMM_NULL, context.get());
        NF_ASSERT(flag == 0, "SUNContext_Create failed");
        context_.swap(context);
    }
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initSUNVector(const Executor& exec, size_t size)
{
    NF_DEBUG_ASSERT(context_, "SUNContext is a nullptr.");
    solution_.setExecutor(exec);
    solution_.initNVector(size, context_);
    initialConditions_.setExecutor(exec);
    initialConditions_.initNVector(size, context_);
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initSUNInitialConditions(const SolutionFieldType& solutionField)
{

    NeoN::sundials::fieldToSunNVector(
        solutionField.internalField(), initialConditions_.sunNVector()
    );
}

template<typename SolutionFieldType>
void RungeKutta<SolutionFieldType>::initODEMemory(const scalar t)
{
    NF_DEBUG_ASSERT(context_, "SUNContext is a nullptr.");
    NF_DEBUG_ASSERT(pdeExpr_, "PDE expression is a nullptr.");

    void* ark = ERKStepCreate(
        NeoN::sundials::explicitRKSolve<SolutionFieldType>,
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
