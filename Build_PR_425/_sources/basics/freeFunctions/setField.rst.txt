.. _basic_functions_setVector:


``setVector``
------------

Header: ``"NeoN/fields/fieldFreeFunctions.hpp"``


Description
^^^^^^^^^^^

The function ``setVector`` sets the entire field with a given field or a subfield with a given field if a range is defined.


Definition
^^^^^^^^^^

.. doxygenfunction:: NeoN::setVector

Example
^^^^^^^

.. code-block:: cpp

    // or any other executor CPUExecutor, SerialExecutor
    NeoN::Executor = NeoN::GPUExecutor{};

    NeoN::Vector<NeoN::scalar> fieldA(exec, 2);
    NeoN::Vector<NeoN::scalar> fieldB(exec, 2, 1.0);
    NeoN::Vector<NeoN::scalar> fieldC(exec, 2, 2.0);
    // Note if the executor does not match the program will exit with a segfault
    NeoN::setVector(field, fieldB.view());
    // only set the last element of the field
    NeoN::map(field, fieldC.view(), {1, 2});
    // copy to host
    auto hostVector = field.copyToHost();
    for (auto i = 0; i < field.size(); ++i)
    {
        std::cout << hostVector[i] << std::endl;
    }
    // prints:
    // 1.0
    // 2.0
