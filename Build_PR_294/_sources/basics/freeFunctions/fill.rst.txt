.. _basic_functions_fill:


``fill``
--------

Header: ``"NeoN/fields/fieldFreeFunctions.hpp"``


Description
^^^^^^^^^^^

The function ``fill`` fills the entire field with a given value or a subfield with a given value if a range is defined.


Definition
^^^^^^^^^^

.. doxygenfunction:: NeoN::fill

Example
^^^^^^^

.. code-block:: cpp

    // or any other executor CPUExecutor, SerialExecutor
    NeoN::Executor = NeoN::GPUExecutor{};

    NeoN::Vector<NeoN::scalar> field(exec, 2);
    NeoN::fill(field, 1.0);
    NeoN::fill(field, 2.0, {1, 2}); // fill a subfield with a value
    // copy to host
    auto hostVector = field.copyToHost();
    for (auto i = 0; i < field.size(); ++i)
    {
        std::cout << hostVector[i] << std::endl;
    }
    // prints:
    // 1.0
    // 2.0
