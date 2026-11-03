.. _basics_unstructuredMesh:

UnstructuredMesh
================

The `unstructuredMesh` holds the relevant data for the representing a computational grid on the selected executor.
However, currently no construction from disc is supported, thus it is mainly used a data container for mesh data.

.. warning::
   - Currently no method to read meshes from disc is implemented. Thus
     mesh data needs to be provided by the user or a converter such as FoamAdapter needs do be used.

Further details `unstructuredMesh  <https://exasim-project.com/NeoN/latest/doxygen/html/classNeoN_1_1UnstructuredMesh.html>`_

BoundaryMesh
^^^^^^^^^^^^

The boundaryMesh in the current implementation stores the relevant data for the `boundaryMesh` on the selected executor.
So it is currently a data container for boundary mesh data.
The `boundaryMesh` information are stored in a continuous array and the index for the boundary patch is stored in a offset.

.. warning::
   - unable to read the boundary mesh from disc
   - boundary mesh data needs to be provided by the user

Further details `boundaryMesh <https://exasim-project.com/NeoN/latest/doxygen/html/classNeoN_1_1BoundaryMesh.html>`_
