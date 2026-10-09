.. _timeIntegration_forward_euler:

Forward Euler
=============

The Forward Euler implementation in NeoN provides a dependency-free first-order explicit time integration method.
It advances the solution from time step n to n+1 using:

.. math::

    y_{n+1} = y_n + \Delta t \cdot f(t_n, y_n)

Implementation
-------------

The implementation is entirely self-contained within NeoN, requiring no external libraries.
The ``ForwardEuler`` class template inherits from ``TimeIntegratorBase`` and implements straightforward time-stepping functionality through its ``solve`` method:

.. code-block:: cpp

    template<typename SolutionVectorType>
    void solve(Expression& eqn, SolutionVectorType& sol, scalar t, scalar dt)
    {
        auto source = eqn.explicitOperation(sol.size());
        sol.internalVector() -= source * dt;
        sol.correctBoundaryConditions();
    }

This lightweight implementation operates directly on NeoN fields and expressions, providing a guaranteed time integration option regardless of build environment constraints.

Usage
-----

To use the Forward Euler integrator, configure it through a dictionary:

.. code-block:: cpp

    Dictionary timeDict;
    timeDict.set("type", "forwardEuler");
    TimeIntegration<VolumeVector<scalar>> integrator(timeDict);

    // Solve for one timestep
    integrator.solve(equation, solutionVector, currentTime, deltaT);

Considerations
-------------

The Forward Euler method serves as both a simple first-order integration option and a fallback when external time integration libraries are unavailable.
While it has minimal computational overhead and no external dependencies, its first-order accuracy means it may require smaller time steps compared to higher-order methods.
For GPU computations, the solver automatically handles execution space synchronization after each time step when required.
