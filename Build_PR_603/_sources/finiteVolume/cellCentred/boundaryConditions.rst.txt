.. _fvcc_BC:

.. warning::
    The API of the classes probably will change in the future. It is a first draft implementation to iterate over the design.


Boundary Conditions
===================

In NeoN the boundary conditions do not store the underlying  data but instead modify the data provided by ``Field``.  A basic NoOp implementation is provided by  the ``VolumeBoundary`` and ``SurfaceBoundary`` classes.
To apply boundary conditions for both Surface and Volume Vectors, a virtual base class member function ``correctBoundaryConditions`` is used. The member acts as an interface and is responsible for updating the actual boundary data contained within an ``InternalVector`` (an attribute of the ``Field``).

.. doxygenclass:: NeoN::finiteVolume::cellCentred::VolumeBoundary
    :members:
        correctBoundaryConditions



.. doxygenclass:: NeoN::finiteVolume::cellCentred::SurfaceBoundary
    :members:
        correctBoundaryConditions

The above are the base classes for the specific (derived) implementations which will ultimately provide the actual boundary conditions to both volumetric and surface fields.

Boundary Conditions for VolumeVector's
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function ``correctBoundaryConditions`` is responsible for modifying the data of the ``boundaryVector`` using the visitor pattern. A possible implementation is shown below.

.. code-block:: c++

    void fvccScalarFixedValueBoundaryVector::correctBoundaryConditions(
        BoundaryData<scalar>& bfield, const Vector<scalar>& internalVector
    )
    {
        fixedValueBCKernel kernel_(mesh_, patchID_, start_, end_, uniformValue_);
        std::visit([&](const auto& exec) { kernel_(exec, bfield, internalVector); }, bfield.exec());
    }

The logic is implemented in the kernel classes:

.. code-block:: c++

    void fixedValueBCKernel::operator()(
        const GPUExecutor& exec, BoundaryData<scalar>& bVector, const Vector<scalar>& internalVector
    )
    {
        using executor = typename GPUExecutor::exec;
        auto s_value = bVector.value().field();
        auto s_refValue = bVector.refValue().field();
        scalar uniformValue = uniformValue_;
        Kokkos::parallel_for(
            "fvccScalarFixedValueBoundaryVector",
            Kokkos::RangePolicy<executor>(start_, end_),
            NEON_LAMBDA(const localIdx i) {
                s_value[i] = uniformValue;
                s_refValue[i] = uniformValue;
            }
        );
    }

As the ``BoundaryData`` class stores all data in a contiguous array, the boundary condition must only update the data in the range of the boundary specified by the `start_` and `end_` index. In the above simple boundary condition, the kernel only sets the values to a uniform/fixed value. The ``value`` field stores the current value of the boundary condition that is used by the explicit operators and the ``refValue`` stores the value of the boundary condition that is used by the implicit operators.

Currently, the following boundary conditions are implemented for volVector for scalar:
- fixedValue
- zeroGradient
- calculated


Slip / Symmetry boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``slip`` and ``symmetry`` share a single implementation. They differ only in their
registered name and in where they may be applied (``slip`` on a wall/regular patch,
``symmetry`` on a symmetry plane). Per face the operator is:

- **scalar field** => zero-gradient (there is no normal component, so the mode below is
  irrelevant).
- **vector field** => the boundary value is the *tangential projection* of the cell value
  (the wall-normal component is removed), plus a **normal-damping** term
  :math:`-\Delta\,(\mathbf{v}\cdot\mathbf{n})\,\mathbf{n}` that drives the normal component
  of the cell value towards zero.

The velocity vector field is assembled as a single shared scalar matrix and solved with a
Vec3 (multi-RHS) solve, so the same matrix diagonal is used for all three components. The
normal-damping coefficient :math:`\gamma\,|S|\,\Delta\,|n_c|` is **direction dependent**
(it carries a per-component :math:`|n_c|` factor) and therefore cannot be held by the shared
scalar diagonal. NeoN offers two ways to realise it, selected per patch by the ``implicit``
key:

Deferred (``implicit no``)
    The damping is written into ``refGrad`` as
    :math:`-\Delta\,(\mathbf{v}\cdot\mathbf{n})\,\mathbf{n}` and enters the per-component
    **RHS** through the existing fixed-gradient assembly. This keeps the shared scalar matrix
    and the fast multi-RHS solve intact. Because the term uses the previous iteration's
    velocity, the normal coupling **lags by one outer iteration**; on a developed field this
    can be too weak to hold the normal component and lets it diverge in the momentum solve.

Implicit (``implicit yes``, the default)
    ``refGrad`` is left zero; instead the ``BoundaryAttributes::transformImplicit`` flag tells
    the Laplacian assembly to accumulate the per-component diagonal correction
    :math:`\gamma\,|S|\,\Delta\,|n_c|` into the linear system's ``diagCmpt`` store. Since this
    correction differs per column, the solver drops the multi-RHS solve and runs the three
    components **segregated**, temporarily subtracting each column's correction from the shared
    diagonal in place (no matrix copy). The constraint is applied *inside* the solve, so it does
    not lag — at the cost of three solves instead of one multi-RHS solve.

The default is **implicit**: the deferred variant was observed to let the wall-normal velocity
run away at slip/symmetry boundaries on developed external-aerodynamics cases. Set
``implicit no`` on a patch to opt back into the deferred, multi-RHS-friendly treatment.

.. note::
    A future ``Tensor`` specialisation (e.g. a transported Reynolds-stress field) must use the
    full reflective transform :math:`(T + H\,T\,H)/2` with :math:`H = I - 2\,\mathbf{n}\otimes\mathbf{n}`,
    **not** the per-component projection used for vectors.
