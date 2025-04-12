Operator
=======


The `Operator` class represents a term in an equation and can be instantiated with different value types.
An `Operator` is either explicit, implicit or temporal, and can be scalable by an additional coefficient, for example a scalar value or a further field.
The `Operator` implementation uses Type Erasure (more details `[1] <https://medium.com/@gealleh/type-erasure-idiom-in-c-0d1cb4f61cf0>`_ `[2] <https://www.youtube.com/watch?v=4eeESJQk-mw>`_ `[3] <https://www.youtube.com/watch?v=qn6OqefuH08>`_) to achieve polymorphism without inheritance. Consequently, the class needs only to implement the interface which is used in the DSL and which is shown in the below example:

Example:
    .. code-block:: cpp

        NeoFOAM::dsl::Operator<NeoFOAM::scalar> divTerm =
            Divergence(NeoFOAM::dsl::Operator<NeoFOAM::scalar>::Type::Explicit, exec, ...);

        NeoFOAM::dsl::Operator<NeoFOAM::scalar> ddtTerm =
            TimeTerm(NeoFOAM::dsl::Operator<NeoFOAM::scalar>::Type::Temporal, exec, ..);


To fit the specification of the Expression (storage in a vector), the Operator needs to be able to be scaled:

.. code-block:: cpp

        NeoFOAM::Field<NeoFOAM::scalar> scalingField(exec, nCells, 2.0);
        auto sF = scalingField.view();

        dsl::Operator<NeoFOAM::scalar> customTerm =
            CustomTerm(dsl::Operator<NeoFOAM::scalar>::Type::Explicit, exec, nCells, 1.0);

        auto constantScaledTerm = 2.0 * customTerm; // A constant scaling factor of 2 for the term.
        auto fieldScaledTerm = scalingField * customTerm; // scalingField is used to scale the term.

        // Operator also supports a similar syntax as OpenFOAM
        auto multiScaledTerm = (scale + scale + scale + scale) * customTerm;

        // Operator also supports the use of a lambda as scaling function to reduce the number of temporaries generated
        auto lambdaScaledTerm =
            (KOKKOS_LAMBDA(const NeoFOAM::size_t i) { return sF[i] + sF[i] + sF[i]  + sF[i]; }) * customTerm;

To add a user-defined `Operator`, a new derived class must be created, inheriting from `OperatorMixin`,
 and provide the definitions of the below virtual functions that are required for the `Operator` interface:

    - build: build the term
    - explicitOperation: perform the explicit operation
    - implicitOperation: perform the implicit operation
    - getType: get the type of the term
    - exec: get the executor
    - volumeField: get the volume field

An example can be found in `test/dsl/operator.cpp`.

The required scaling of the term is handled by the `Coeff` type which can be retrieved by the `getCoefficient` function of `Operator`.
