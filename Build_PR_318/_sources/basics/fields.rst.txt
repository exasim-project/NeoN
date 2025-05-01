.. _fvcc_fields:
Vectors
======

Overview
^^^^^^^^

.. warning::
    The API of the classes probably will change in the future as currently parallelization is not supported.

NeoN implements several field classes:

- ``Vector<ValueType>`` the basic GPU capable container class supporting algebraic operations
- ``BoundaryData<ValueType>`` A GPU friendly datastructure storing boundary data.
- ``Field<ValueType>`` The combination of an internal field and its corresponding boundary data.

Besides these container like field classes several finite volume specific field classes are implemented. The corresponding classes are:

- ``DomainMixin<ValueType>`` Mixin class combining a ``Field`` and the corresponding mesh.
- ``VolumeVector<ValueType>`` Uses the DomainMixin and implements finite volume specific members, including the notion of concrete boundary condiditons
- ``SurfaceVector<ValueType>`` The surface field equivalent to ``VolumeVector``

The Vector<ValueType> class
^^^^^^^^^^^^^^^^^^^^^^^^^^
The Vector class is the basic container class and is the central component for implementing a platform portable CFD framework.
One of the key differences between accessing the elements of a ``Vector`` and typical ``std`` sequential data containers is the lack of subscript or direct element access operators.
This is to prevent accidental access to device memory from the host.
The correct procedure to access ``Vector`` elements is indirectly through a ``view``, as shown below:
.. code-block:: cpp

    // Host Vectors
    Vector<T> hostVector(Executor::CPUExecutor, size_);
    auto hostVectorView = hostVector.view();
    hostVectorView[1] = 1; // assuming size_ > 2.

    // Device Vectors
    Vector<T> deviceVector(Executor::GPUExecutor, size_);
    auto deviceVectorOnHost = deviceVector.copyToHost();
    auto deviceVectorOnHostView = deviceVectorOnHost.view();
    deviceVectorOnHostView[1] = 1; // assuming size_ > 2.

Vectors support basic algebraic operations such as binary operations like the addition or subtraction of two fields, or scalar operations like the multiplication of a field with a scalar.
In the following, some implementation details of the field operations are detailed using the additions operator as an example.
The block of code below shows an example implementation of the addition operator.

.. code-block:: cpp

    [[nodiscard]] Vector<T> operator+(const Vector<T>& rhs)
    {
        Vector<T> result(exec_, size_);
        result = *this;
        add(result, rhs);
        return result;
    }


Besides creating a temporary for the result it mainly calls the free standing ``add`` function which is implemented in ``include/NeoN/field/fieldFreeFunctions.hpp``.
This, in turn, dispatches to the ``fieldBinaryOp`` function, passing the actual kernel as lambda.
The ``fieldBinaryOp``  is implemented using our parallelFor implementations which ultimately dispatch to the ``Kokkos::parallel_for`` function, see `Kokkos documentation  <https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_for.html>`_ for more details.

.. code-block:: cpp

    template<typename ValueType>
    void add(Vector<ValueType>& a, const Vector<std::type_identity_t<ValueType>>& b)
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
                KOKKOS_LAMBDA(const localIdx i) { kernel(i); }
            );
        }
    }

The code snippet highlights another important aspect, the executor.
The executor defines the ``Kokkos::RangePolicy``, see  `Kokkos Programming Model  <https://github.com/kokkos/kokkos-core-wiki/blob/main/docs/source/ProgrammingGuide/ProgrammingModel.md>`_.
Besides defining the RangePolicy, the executor also holds functions for allocating and deallocationg memory.
See our `documentation  <https://exasim-project.com/NeoN/latest/basics/executor.html>`_ for more details on the executor model.

Further `Details  <https://exasim-project.com/NeoN/latest/doxygen/html/classNeoN_1_1Vector.html>`_.

Cell Centred Specific Vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within in the ``finiteVolume/cellCentred`` folder and the namespace
``NeoN::finiteVolume::cellCentred`` two specific field types, namely the ``VolumeVector`` and the ``SurfaceVector`` are implemented.
Both derive from the ``DomainMixin`` a mixin class which handles that all derived fields contain geometric information via the mesh data member and field specific data via the ``Field`` data member.

``Field`` acts as the fundamental data container within this structure, offering both read and write to the ``internalVector`` and  ``boundaryVectors`` data structure holding actual boundary data.

The ``VolumeVector`` and the ``SurfaceVector`` hold a vector of boundary conditions implemented in ``finiteVolume/cellCentred/boundary`` and a  ``correctBoundaryConditions`` member function that updates the field's boundary condition.

The SurfaceVector implementation of ``internalVector`` also contains the boundary values, so no branches (if) are required when iterating over all cell faces.

Further details `VolumeVector  <https://exasim-project.com/NeoN/latest/doxygen/html/classNeoN_1_1finiteVolume_1_1cellCentred_1_1VolumeVector.html>`_ and `ScalarVector  <https://exasim-project.com/NeoN/latest/doxygen/html/classNeoN_1_1finiteVolume_1_1cellCentred_1_1ScalarVector.html>`_.

.. _api_fields:
