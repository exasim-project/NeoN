.. _basic_functions_setField:


``setField``
------------

Header: ``"NeoFOAM/fields/fieldFreeFunctions.hpp"``


Description
^^^^^^^^^^^

The function ``setField`` sets the entire field with a given field or a subfield with a given field if a range is defined.


Definition
^^^^^^^^^^

.. doxygenfunction:: NeoFOAM::setField

Example
^^^^^^^

.. code-block:: cpp

    // or any other executor CPUExecutor, SerialExecutor
    NeoFOAM::Executor = NeoFOAM::GPUExecutor{};

    NeoFOAM::Field<NeoFOAM::scalar> fieldA(exec, 2);
    NeoFOAM::Field<NeoFOAM::scalar> fieldB(exec, 2, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> fieldC(exec, 2, 2.0);
    // Note if the executor does not match the program will exit with a segfault
    NeoFOAM::setField(field, fieldB.view());
    // only set the last element of the field
    NeoFOAM::map(field, fieldC.view(), {1, 2});
    // copy to host
    auto hostField = field.copyToHost();
    for (auto i = 0; i < field.size(); ++i)
    {
        std::cout << hostField[i] << std::endl;
    }
    // prints:
    // 1.0
    // 2.0
