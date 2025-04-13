.. _timeIntegration_index:


Time Integration
================

NeoN provides flexible time integration capabilities for solving partial differential equations (PDEs).
Currently the framework implements two distinct approaches to time integration - a native Forward Euler method and advanced Runge-Kutta methods via Sundials integration.

The native Forward Euler implementation serves two key purposes: it provides a lightweight, dependency-free option for basic time integration, and it acts as a fallback when Sundials is not available in the build environment.
This implementation directly integrates with NeoN's field operations and requires no external libraries.

For more demanding and flexible applications, the Sundials-based Runge-Kutta implementation offers higher-order explicit time integration with Kokkos support.
This integration combines the robustness of Sundials' time integration capabilities with NeoN's field operations.

Both approaches are implemented through the common ``TimeIntegratorBase`` interface, allowing seamless switching between methods through runtime configuration:

NOTE: Native Forward Euler is WIP.

.. code-block:: cpp

    Dictionary timeDict;
    timeDict.set("type", "forwardEuler");  // or "Runge-Kutta"
    TimeIntegration<VolumeVector<scalar>> integrator(time
The following sections provide detailed information about each implementation:

.. toctree::
   :maxdepth: 2
   :glob:

   forwardEuler.rst
   rungeKutta.rst
