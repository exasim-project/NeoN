.. _fvcc_segmentedVectors:

SegmentedVector
^^^^^^^^^^^^^^

SegmentedVector is a template class that represents a field divided into multiple segments and can represent vector of vector of a defined ValueType.
It also allows the definition of an IndexType, so each segment of the vector can be addressed.
It can be used to represent cell to cell stencil.

.. code-block:: cpp

    NeoN::Vector<NeoN::label> values(exec, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    NeoN::Vector<NeoN::localIdx> segments(exec, {0, 2, 4, 6, 8, 10});

    NeoN::SegmentedVector<NeoN::label, NeoN::localIdx> segVector(values, segments);
    auto [valueView, segment] = segVector.views();
    auto segView = segVector.view();
    NeoN::Vector<NeoN::label> result(exec, 5);

    NeoN::fill(result, 0);
    auto resultView = result.view();

    parallelFor(
        exec,
        {0, segVector.numSegments()},
        KOKKOS_LAMBDA(const localIdx segI) {
            // check if it works with bounds
            auto [bStart, bEnd] = segView.bounds(segI);
            auto bVals = valueView.subview(bStart, bEnd - bStart);
            for (auto& val : bVals)
            {
                resultView[segI] += val;
            }

            // check if it works with range
            auto [rStart, rLength] = segView.range(segI);
            auto rVals = valueView.subview(rStart, rLength);
            for (auto& val : rVals)
            {
                resultView[segI] += val;
            }

            // check with subview
            auto vals = segView.view(segI);
            for (auto& val : vals)
            {
                resultView[segI] += val;
            }
        }
    );

In this example, each of the five segments would have a size of two.
This data allows the representation of stencils in a continuous memory layout, which can be beneficial for performance optimization in numerical simulations especially on GPUs.

The views method return the value and segment view and it is also possible to return a view that can also be called on a device
