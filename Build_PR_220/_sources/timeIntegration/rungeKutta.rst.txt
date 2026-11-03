.. _timeIntegration_rungeKutta:

Runge Kutta
===========

The Runge-Kutta implementation in NeoFOAM leverages the Sundials library to provide high-order explicit time integration methods.
This integration combines the robustness of Sundials with NeoFOAM's field operations and Kokkos-based parallel execution capabilities.

Implementation
-------------

The ``RungeKutta`` class template provides a RAII-based wrapper around Sundials' ERKStep module, managing all necessary resources for time integration.
The implementation handles the conversion between NeoFOAM's Kokkos-based fields and Sundials' vector representations:

.. code-block:: cpp

    template<typename SolutionFieldType>
    void solve(Expression& exp, SolutionFieldType& solutionField, scalar t, const scalar dt)
    {
        if (pdeExpr_ == nullptr) initSUNERKSolver(exp, solutionField, t);
        // ... time integration logic
    }

The class manages several key components:
- Sundials context and memory management through RAII
- Conversion between NeoFOAM fields and Sundials vectors
- Integration with Kokkos execution spaces
- Configuration of specific Runge-Kutta methods

Usage
-----

Configure and use the Runge-Kutta integrator through a dictionary:

.. code-block:: cpp

    Dictionary timeDict;
    timeDict.set("type", "Runge-Kutta");
    timeDict.set("Runge-Kutta Method", "Forward Euler");  // Choose RK method
    TimeIntegration<VolumeField<scalar>> integrator(timeDict);

    // Solve for one timestep
    integrator.solve(equation, solutionField, currentTime, deltaT);
