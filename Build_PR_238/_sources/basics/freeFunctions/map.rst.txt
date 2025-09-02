.. _basic_functions_map:


``map``
-------

Header: ``"NeoN/fields/fieldFreeFunctions.hpp"``


Description
^^^^^^^^^^^

The function ``map`` applies a function to each element of the field or a subfield if a range is defined.


Definition
^^^^^^^^^^

.. doxygenfunction:: NeoN::map

Example
^^^^^^^

.. code-block:: cpp

    // or any other executor CPUExecutor, SerialExecutor
    NeoN::Executor = NeoN::GPUExecutor{};

    NeoN::Vector<NeoN::scalar> field(exec, 2);
    NeoN::map(field, KOKKOS_LAMBDA(const std::size_t i) { return 1.0; });
    NeoN::map(field, KOKKOS_LAMBDA(const std::size_t i) { return 2.0; }, {1, 2}); // apply a function to a subfield
    // copy to host
    auto hostVector = field.copyToHost();
    for (auto i = 0; i < field.size(); ++i)
    {
        std::cout << hostVector[i] << std::endl;
    }
    // prints:
    // 1.0
    // 2.0
