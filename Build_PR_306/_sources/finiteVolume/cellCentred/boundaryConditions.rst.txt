.. _fvcc_BC:

.. warning::
    The API of the classes probably will change in the future. It is a first draft implementation to iterate over the design.


Boundary Conditions
===================

In contrast to OpenFOAM, the boundary conditions do not store the underlying  data but instead modify the data provided by ``Field``.  A basic NoOp implementation is provided by  the ``VolumeBoundary`` and ``SurfaceBoundary`` classes.
To apply boundary conditions for both Surface and Volume Vectors, a virtual base class member function ``correctBoundaryConditions`` is used. The member acts as an interface and is responsible for updating the actual boundary data contained within an ``InternalVector`` (an attribute of the ``Field``).

.. doxygenclass:: NeoN::finiteVolume::cellCentred::VolumeBoundary
    :members:
        correctBoundaryConditions



.. doxygenclass:: NeoN::finiteVolume::cellCentred::SurfaceBoundary
    :members:
        correctBoundaryConditions

The above are the base classes for the specific (derived) implementations which will ultimately provide the actual boundary conditions to both volumetric and surface fields.

Boundary Conditions for VolumeVector's
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function ``correctBoundaryConditions`` is responsible for modifying the data of the ``boundaryVector`` using the visitor pattern. A possible implementation is shown below.

.. code-block:: c++

    void fvccScalarFixedValueBoundaryVector::correctBoundaryConditions(
        BoundaryData<scalar>& bfield, const Vector<scalar>& internalVector
    )
    {
        fixedValueBCKernel kernel_(mesh_, patchID_, start_, end_, uniformValue_);
        std::visit([&](const auto& exec) { kernel_(exec, bfield, internalVector); }, bfield.exec());
    }

The logic is implemented in the kernel classes:

.. code-block:: c++

    void fixedValueBCKernel::operator()(
        const GPUExecutor& exec, BoundaryData<scalar>& bVector, const Vector<scalar>& internalVector
    )
    {
        using executor = typename GPUExecutor::exec;
        auto s_value = bVector.value().field();
        auto s_refValue = bVector.refValue().field();
        scalar uniformValue = uniformValue_;
        Kokkos::parallel_for(
            "fvccScalarFixedValueBoundaryVector",
            Kokkos::RangePolicy<executor>(start_, end_),
            KOKKOS_LAMBDA(const localIdx i) {
                s_value[i] = uniformValue;
                s_refValue[i] = uniformValue;
            }
        );
    }

As the ``BoundaryData`` class stores all data in a contiguous array, the boundary condition must only update the data in the range of the boundary specified by the `start_` and `end_` index. In the above simple boundary condition, the kernel only sets the values to a uniform/fixed value. The ``value`` field stores the current value of the boundary condition that is used by the explicit operators and the ``refValue`` stores the value of the boundary condition that is used by the implicit operators.

Currently, the following boundary conditions are implemented for volVector for scalar:
- fixedValue
- zeroGradient
- calculated
