.. _fvcc_fields:
Fields
======

Overview
^^^^^^^^

.. warning::
    The API of the classes probably will change in the future as currently parallelization is not supported.

NeoFOAM implements several field classes:

- ``Field<ValueType>`` the basic GPU capable container class supporting algebraic operations
- ``BoundaryFields<ValueType>`` A GPU friendly datastructure storing boundary data.
- ``DomainField<ValueType>`` The combination of an internal field and its corresponding boundary data.

Besides these container like field classes several finite volume specific field classes are implemented. The corresponding classes are:

- ``GeometricFieldMixin<ValueType>`` Mixin class combining a ``DomainField`` and the corresponding mesh.
- ``VolumeField<ValueType>`` Uses the GeometricFieldMixin and implements finite volume specific members, including the notion of concrete boundary condiditons
- ``SurfaceField<ValueType>`` The surface field equivalent to ``VolumeField``

The Field<ValueType> class
^^^^^^^^^^^^^^^^^^^^^^^^^^
The Field class is the basic container class and is the central component for implementing a platform portable CFD framework.
One of the key differences between accessing the elements of a ``Field`` and typical ``std`` sequential data containers is the lack of subscript or direct element access operators.
This is to prevent accidental access to device memory from the host.
The correct procedure to access ``Field`` elements is indirectly through a ``view``, as shown below:
.. code-block:: cpp

    // Host Fields
    Field<T> hostField(Executor::CPUExecutor, size_);
    auto hostFieldView = hostField.view();
    hostFieldView[1] = 1; // assuming size_ > 2.

    // Device Fields
    Field<T> deviceField(Executor::GPUExecutor, size_);
    auto deviceFieldOnHost = deviceField.copyToHost();
    auto deviceFieldOnHostView = deviceFieldOnHost.view();
    deviceFieldOnHostView[1] = 1; // assuming size_ > 2.

Fields support basic algebraic operations such as binary operations like the addition or subtraction of two fields, or scalar operations like the multiplication of a field with a scalar.
In the following, some implementation details of the field operations are detailed using the additions operator as an example.
The block of code below shows an example implementation of the addition operator.

.. code-block:: cpp

    [[nodiscard]] Field<T> operator+(const Field<T>& rhs)
    {
        Field<T> result(exec_, size_);
        result = *this;
        add(result, rhs);
        return result;
    }


Besides creating a temporary for the result it mainly calls the free standing ``add`` function which is implemented in ``include/NeoFOAM/field/fieldFreeFunctions.hpp``.
This, in turn, dispatches to the ``fieldBinaryOp`` function, passing the actual kernel as lambda.
The ``fieldBinaryOp``  is implemented using our parallelFor implementations which ultimately dispatch to the ``Kokkos::parallel_for`` function, see `Kokkos documentation  <https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_for.html>`_ for more details.

.. code-block:: cpp

    template<typename ValueType>
    void add(Field<ValueType>& a, const Field<std::type_identity_t<ValueType>>& b)
    {
      detail::fieldBinaryOp(
          a, b, KOKKOS_LAMBDA(ValueType va, ValueType vb) { return va + vb; }
      );
    }

A simplified version of the ``parallelFor`` function is shown below.

.. code-block:: cpp
    template<typename Executor, parallelForKernel Kernel>
    void parallelFor(
        [[maybe_unused]] const Executor& exec, std::pair<size_t, size_t> range, Kernel kernel
    )
    {
        auto [start, end] = range;
        if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
        {
        ...
        }
        else
        {
            using runOn = typename Executor::exec;
            Kokkos::parallel_for(
                "parallelFor",
                Kokkos::RangePolicy<runOn>(start, end),
                KOKKOS_LAMBDA(const size_t i) { kernel(i); }
            );
        }
    }

The code snippet highlights another important aspect, the executor.
The executor defines the ``Kokkos::RangePolicy``, see  `Kokkos Programming Model  <https://github.com/kokkos/kokkos-core-wiki/blob/main/docs/source/ProgrammingGuide/ProgrammingModel.md>`_.
Besides defining the RangePolicy, the executor also holds functions for allocating and deallocationg memory.
See our `documentation  <https://exasim-project.com/NeoFOAM/latest/basics/executor.html>`_ for more details on the executor model.

Further `Details  <https://exasim-project.com/NeoFOAM/latest/doxygen/html/classNeoFOAM_1_1Field.html>`_.

Cell Centred Specific Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within in the ``finiteVolume/cellCentred`` folder and the namespace
``NeoFOAM::finiteVolume::cellCentred`` two specific field types, namely the ``VolumeField`` and the ``SurfaceField`` are implemented.
Both derive from the ``GeometricFieldMixin`` a mixin class which handles that all derived fields contain geometric information via the mesh data member and field specific data via the ``DomainField`` data member.

``DomainField`` acts as the fundamental data container within this structure, offering both read and write to the ``internalField`` and  ``boundaryFields`` data structure holding actual boundary data.

The ``VolumeField`` and the ``SurfaceField`` hold a vector of boundary conditions implemented in ``finiteVolume/cellCentred/boundary`` and a  ``correctBoundaryConditions`` member function that updates the field's boundary condition.

Functionally, the ``VolumeField`` and the ``SurfaceField`` classes are comparable to OpenFOAM classes such as ``volScalarField``, ``volVectorField``, and ``volTensorField`` or ``surfaceScalarField``, ``surfaceVectorField``, and ``surfaceTensorField`` respectively.

A difference in the SurfaceField implementation is that the ``internalField`` also contains the boundary values, so no branches (if) are required when iterating over all cell faces.
Thus the size of the ``internalField`` in NeoFOAM differs from that of OpenFOAM.

Further details `VolumeField  <https://exasim-project.com/NeoFOAM/latest/doxygen/html/classNeoFOAM_1_1finiteVolume_1_1cellCentred_1_1VolumeField.html>`_ and `ScalarField  <https://exasim-project.com/NeoFOAM/latest/doxygen/html/classNeoFOAM_1_1finiteVolume_1_1cellCentred_1_1ScalarField.html>`_.

.. _api_fields:
