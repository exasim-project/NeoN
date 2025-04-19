.. _algorithms:

Parallel Algorithms
===================

Parallel algorithms are a basic building block for implementing advanced kernels.
To simplify the implementation of the advanced kernels we provide a set of standard algorithms.
These can be found in the following files:

- ``include/NeoN/core/parallelAlgorithms.hpp``
- ``test/core/parallelAlgorithms.cpp``

Currently, the following algorithms are provided:

- ``parallelFor``
- ``parallelReduce``

The following code block shows the implementation of a parallelFor for fields

.. code-block:: cpp

    template<typename Executor, typename ValueType, parallelForVectorKernel<ValueType> Kernel>
    void parallelFor([[maybe_unused]] const Executor& exec, Vector<ValueType>& field, Kernel kernel)
    {
        auto view = field.view();
        if constexpr (std::is_same<std::remove_reference_t<Executor>, SerialExecutor>::value)
        {
            for (size_t i = 0; i < field.size(); i++)
            {
                view[i] = kernel(i);
            }
        }
        else
        {
            using runOn = typename Executor::exec;
            Kokkos::parallel_for(
                "parallelFor",
                Kokkos::RangePolicy<runOn>(0, field.size()),
                KOKKOS_LAMBDA(const localIdx i) { view[i] = kernel(i); }
            );
        }
    }


based on the Executor type a kernel function is either run directly within a for loop or dispatched to ``Kokkos::parallel_for`` for all non ``SerialExecutors``.
The executor type determines the ``Kokkos::RangePolicy<runOn>`` and thus dispatches to GPUs if a ``GPUExecutor`` was used.
Additionally, we name the kernel as ``"parallelFor"`` to improve visibility in profiling tools like nsys.
Finally, a ``KOKKOS_LAMBDA`` is dispatched assigning the result of the given kernel function to the view of the field.
Here the view holds data pointers to the device data and defines the begin and end pointer of the data.
Several overloads of the ``parallelFor`` functions exists to simplify running parallelFor on fields and spans with and without an explicitly defined data range.


To learn more on how to use the algorithms it is recommended to check the corresponding `unit test <https://github.com/exasim-project/NeoN/blob/main/test/core/parallelAlgorithms.cpp>`_.

Further details `parallelFor <https://exasim-project.com/NeoN/latest/doxygen/html/parallelAlgorithms_8hpp_source.html>`_.

Currently, the following free functions are implemented:

.. toctree::
   :maxdepth: 1

   ./freeFunctions/fill.rst
   ./freeFunctions/map.rst
   ./freeFunctions/setVector.rst
