.. _basic_functions_map:


``map``
-------

Header: ``"NeoFOAM/fields/fieldFreeFunctions.hpp"``


Description
^^^^^^^^^^^

The function ``map`` applies a function to each element of the field or a subfield if a range is defined.


Definition
^^^^^^^^^^

.. doxygenfunction:: NeoFOAM::map

Example
^^^^^^^

.. code-block:: cpp

    // or any other executor CPUExecutor, SerialExecutor
    NeoFOAM::Executor = NeoFOAM::GPUExecutor{};

    NeoFOAM::Field<NeoFOAM::scalar> field(exec, 2);
    NeoFOAM::map(field, KOKKOS_LAMBDA(const std::size_t i) { return 1.0; });
    NeoFOAM::map(field, KOKKOS_LAMBDA(const std::size_t i) { return 2.0; }, {1, 2}); // apply a function to a subfield
    // copy to host
    auto hostField = field.copyToHost();
    for (auto i = 0; i < field.size(); ++i)
    {
        std::cout << hostField[i] << std::endl;
    }
    // prints:
    // 1.0
    // 2.0
