.. _fvcc:

Domain Specific Language (DSL)
==============================

The concept of a Domain Specific Language (DSL) allows to simplify the process of implementing and solving equations in a given programming language like C++.
Engineers can express equations in a concise and readable form, closely resembling their mathematical representation, while no or little knowledge of the numerical schemes and implementation is required.
This approach allows engineers to focus on the physics of the problem rather than the numerical implementation and helps in reducing the time and effort required to develop and maintain complex simulations.

The Navier-Stokes equations can be expressed in the DSL in OpenFOAM as follows:

.. code-block:: cpp

    fvVectorMatrix UEqn
    (
        fvm::ddt(U)
        + fvm::div(phi, U)
        - fvm::laplacian(nu, U)
    );

    solve(UEqn == -fvc::grad(p));

To solve the continuity equation in OpenFOAM with the PISO or SIMPLE algorithm, the VectorMatrix, UEqn, needs to provide the diagonal and off-diagonal terms of the matrix.

.. code-block:: cpp

    volScalarField rAU(1.0/UEqn.A());
    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

This approach is readable and easy to understand for engineers familiar with OpenFOAM. However, it has several limitations due to its implementation in OpenFOAM:

    - the solution system is always a sparse matrix
    - individual terms are eagerly evaluated, resulting in unnecessary use of temporaries
    - the sparse matrix is always an LDU matrix that is not supported by external linear solvers
    - Only cell-centred discretisation is supported

NeoN DSL tries to address these issues by providing:

    - lazy evaluation of the equations terms. This allows for better optimisation of the resulting equation and can reduce the number of required temporaries.
    - a more modular design
    - Support for standard matrix formats like COO and CSR, to simplify the use of external linear solvers

The use of standard matrix formats combined with lazy evaluation allows for the use of external libraries to integrate PDEs in time and space.
The equation system can be passed to **sundials** and be integrated by **RK methods** and **BDF methods** on heterogeneous architectures.

The NeoN DSL is designed as drop in replacement for OpenFOAM DSL and the adoption should be possible with minimal effort and the same equation from above should read:

.. code-block:: cpp

    dsl::Equation<NeoN::scalar> UEqn
    (
        dsl::imp::ddt(U)
        + dsl::imp::div(phi, U)
        - dsl::imp::laplacian(nu, U)
    )

    solve(UEqn == -dsl::exp::grad(p), U, fvSchemes, fvSolution);


In contrast to OpenFOAM, the matrix assembly is deferred till the solve step. Hence the majority of the computational work is performed during the solve step.
That is 1. assemble the system and 2. solve the system.
After the system is assembled or solved, it provides access to the linear system for the SIMPLE and PISO algorithms.



.. toctree::
    :maxdepth: 2
    :glob:

    equation.rst
    operator.rst
