.. _first_kernel:

Case Study: The Gauss Green Div Kernel
======================================

This section explains discusses how the Gauss Green Div Kernel is implemented.
The class is implemented in the following files:

- ``include/NeoN/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp``
- ``src/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp``
- ``test/finiteVolume/cellCentred/operators/gaussGreenDiv.hpp``

The ``gaussGreenDiv`` class represents the following term :math:`\int \nabla \cdot \phi dV` and is a particular implementation of the ``dsl::explicit::div`` operator.
Hence, in order to make the implementation selectable at runtime we let the ``GaussGreenDiv`` class derive from ``DivOperatorFactory::Register<GaussGreenDiv>`` and implement the static name function, (see also registerClass).

The actual implementation of the operator can be found in the ``gaussGreenDiv.cpp`` file.
The ``GaussGreenDiv::div`` member calls a free standing function ``computeDiv`` with the correct arguments.
In NeoN it is a common pattern to use free standing functions since they are easier to test and communicate all dependencies explicitly via the function arguments.


The discretized version of the divergence term can be written as

.. math::

   \int \nabla \phi dV = \int dS\cdot\phi = \sum_f S_f\cdot\phi_f


and the internal field part is implemented in OpenFOAM as

.. code-block:: cpp

    forAll(owner, facei)
    {
        const GradType Sfssf = Sf[facei]*issf[facei];

        igGrad[owner[facei]] += Sfssf;
        igGrad[neighbour[facei]] -= Sfssf;
    }


the corresponding NeoN version is implemented as

.. code-block:: cpp

    parallelFor(
        exec,
        {0, nInternalFaces},
        KOKKOS_LAMBDA(const size_t i) {
            scalar flux = surfFaceFlux[i]*surfPhif[i];
            threadsafe_add(&surfDivPhi[static_cast<size_t>(surfOwner[i])], flux);
            threadsafe_sub(&surfDivPhi[static_cast<size_t>(surfNeighbour[i])], flux);
        }
    );


Here, the following changes have been applied

- replace the ``forAll`` macro with the ``NeoN::parallelFor`` (see `parallelFor <https://exasim-project.com/NeoN/Build_PR_221/basics/algorithms.html>`_.) function which takes the executor, the range, and the loop body as arguments.
- calls to ``+=`` and ``-=`` are replaced by the ``threadsafe_add`` and ``threadsafe_sub`` function which takes the lhs and rhs as arguments.
