.. _basic_functions_fill:


``fill``
--------

Header: ``"NeoFOAM/fields/fieldFreeFunctions.hpp"``


Description
^^^^^^^^^^^

The function ``fill`` fills the entire field with a given value or a subfield with a given value if a range is defined.


Definition
^^^^^^^^^^

.. doxygenfunction:: NeoFOAM::fill

Example
^^^^^^^

.. code-block:: cpp

    // or any other executor CPUExecutor, SerialExecutor
    NeoFOAM::Executor = NeoFOAM::GPUExecutor{};

    NeoFOAM::Field<NeoFOAM::scalar> field(exec, 2);
    NeoFOAM::fill(field, 1.0);
    NeoFOAM::fill(field, 2.0, {1, 2}); // fill a subfield with a value
    // copy to host
    auto hostField = field.copyToHost();
    for (auto i = 0; i < field.size(); ++i)
    {
        std::cout << hostField[i] << std::endl;
    }
    // prints:
    // 1.0
    // 2.0
