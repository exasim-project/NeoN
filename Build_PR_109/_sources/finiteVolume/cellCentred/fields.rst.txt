.. _fvcc_fields:

Fields (FVCC)
=============

.. warning::
    The API of the classes probably will change in the future as currently parallelization is not supported.

Cell Centered Fields
^^^^^^^^^^^^^^^^^^^^

The ``VolumeField`` stores the field values at cell centers and along boundaries, providing essential data for constructing the DSL (Domain Specific Language). This functionality also includes access to mesh data, integrating closely with the computational framework.

``DomainField`` acts as the fundamental data container within this structure, offering both read and write to the ``internalField`` and ``boundaryFields`` provided by the ``DomainField``. The ``correctBoundaryConditions`` member function updates the field's boundary conditions, which are specified at construction. The boundaryConditions do not hold the data but rather modify the ``DomainField`` or ``BoundaryField`` container.

Functionally, fvccVolField parallels several OpenFOAM classes such as ``volScalarField``, ``volVectorField``, and ``volTensorField``.

.. doxygenclass:: NeoFOAM::fvccVolField
    :members:
        field_,
        fvccVolField,
        internalField,
        boundaryField,
        correctBoundaryConditions

Face Centered fields
^^^^^^^^^^^^^^^^^^^^

The ``SurfaceField`` class stores the field values interpreted as face centers values.  Additionally it stores boundaries the corresponding boundary conditions. This provides essential data for constructing the DSL (Domain Specific Language). The functionality also includes access to mesh data, integrating closely with the computational framework.

``DomainField`` acts as the fundamental data container within this structure, offering both read and to the ``internalField`` and ``boundaryField`` provided by the ``DomainField``. The ``correctBoundaryConditions`` member function updates field's boundary conditions, which are specified at construction. The boundaryConditions do not hold the data but modify the ``DomainField`` or ``BoundaryField`` container.

Functionally, fvccVolField parallels several OpenFOAM classes such as ``surfaceScalarField``, ``surfaceVectorField``, and ``surfaceTensorField``.
However, the internalField also contains the boundary values, so no branches (if) are required when iterating over all cell faces. Thus the size of the internalField in NeoFOAM differs from that of OpenFOAM.

.. doxygenclass:: NeoFOAM::fvccSurfaceField
    :members:
        field_,
        fvccSurfaceField,
        internalField,
        boundaryField,
        correctBoundaryConditions
