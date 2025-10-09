.. _fvcc:

Domain Specific Language (DSL)
==============================

The concept of a Domain Specific Language (DSL) allows to simplify the process of implementing and solving equations in a given programming language like C++.
Engineers can express equations in a concise and readable form, closely resembling their mathematical representation, while no or little knowledge of the numerical schemes and implementation is required.
This approach allows engineers to focus on the physics of the problem rather than the numerical implementation and helps in reducing the time and effort required to develop and maintain complex simulations.

The use of standard matrix formats combined with lazy evaluation allows for the use of external libraries to integrate PDEs in time and space.
The equation system can be passed to **sundials** and be integrated by **RK methods** and **BDF methods** on heterogeneous architectures.

With the NeoN DSL the implementation of a momentum equation should read

.. code-block:: cpp

    dsl::Expression<NeoN::scalar> momentum
    (
        dsl::imp::ddt(U)
        + dsl::imp::div(phi, U)
        - dsl::imp::laplacian(nu, U)
    )

    solve(momentum == -dsl::exp::grad(p), U, solverProperties);


In contrast to other CFD software, the matrix assembly is deferred till the solve step. Hence the majority of the computational work is performed during the solve step.
That is 1. assemble the system and 2. solve the system.
After the system is assembled or solved, it provides access to the linear system for the SIMPLE and PISO algorithms.


.. toctree::
    :maxdepth: 2
    :glob:

    equation.rst
    operator.rst
