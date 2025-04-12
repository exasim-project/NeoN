.. _fvcc_segmentedFields:

SegmentedField
^^^^^^^^^^^^^^

SegmentedField is a template class that represents a field divided into multiple segments and can represent vector of vector of a defined ValueType.
It also allows the definition of an IndexType, so each segment of the vector can be addressed.
It can be used to represent cell to cell stencil.

.. code-block:: cpp

    NeoFOAM::Field<NeoFOAM::label> values(exec, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    NeoFOAM::Field<NeoFOAM::localIdx> segments(exec, {0, 2, 4, 6, 8, 10});

    NeoFOAM::SegmentedField<NeoFOAM::label, NeoFOAM::localIdx> segField(values, segments);
    auto [valueSpan, segment] = segField.spans();
    auto segView = segField.view();
    NeoFOAM::Field<NeoFOAM::label> result(exec, 5);

    NeoFOAM::fill(result, 0);
    auto resultSpan = result.view();

    parallelFor(
        exec,
        {0, segField.numSegments()},
        KOKKOS_LAMBDA(const size_t segI) {
            // check if it works with bounds
            auto [bStart, bEnd] = segView.bounds(segI);
            auto bVals = valueSpan.subspan(bStart, bEnd - bStart);
            for (auto& val : bVals)
            {
                resultSpan[segI] += val;
            }

            // check if it works with range
            auto [rStart, rLength] = segView.range(segI);
            auto rVals = valueSpan.subspan(rStart, rLength);
            for (auto& val : rVals)
            {
                resultSpan[segI] += val;
            }

            // check with subspan
            auto vals = segView.span(segI);
            for (auto& val : vals)
            {
                resultSpan[segI] += val;
            }
        }
    );

In this example, each of the five segments would have a size of two.
This data allows the representation of stencils in a continuous memory layout, which can be beneficial for performance optimization in numerical simulations especially on GPUs.

The spans method return the value and segment span and it is also possible to return a view that can also be called on a device
