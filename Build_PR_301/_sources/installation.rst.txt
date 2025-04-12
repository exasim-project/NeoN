Installation
============

You can build NeoFOAM by following these steps:

Clone the NeoFOAM repository:

   .. code-block:: bash

      git clone https://github.com/exasim-project/NeoFOAM.git

Navigate to the NeoFOAM directory:

   .. code-block:: bash

      cd NeoFOAM

NeoFOAM uses CMake to build, thus the standard CMake procedure should work, however, we recommend using one of the provided CMake presets detailed below `below <Building with CMake Presets>`_. From a build directory, you can execute:

   .. code-block:: bash

        mkdir build
        cd build
        cmake <DesiredBuildFlags> ..
        cmake --build .
        cmake --install .

The following can be chained with -D<DesiredBuildFlags>=<Value> to the CMake command most and most relevant build flags are:

+---------------------------+-----------------------------------+---------+
| Flag                      | Description                       | Default |
+===========================+===================================+=========+
| CMAKE_BUILD_TYPE          | Build in debug or release mode    | Debug   |
+---------------------------+-----------------------------------+---------+
| NEOFOAM_BUILD_DOC         | Build NeoFOAM with documentation  | ON      |
+---------------------------+-----------------------------------+---------+
| NEOFOAM_BUILD_TESTS       | Build NeoFOAM with tests          | OFF     |
+---------------------------+-----------------------------------+---------+

To browse the full list of build options it is recommended to use a build tool like ``ccmake``.
By opening the the project with cmake-gui you can easily set these flags and configure the build.
NeoFOAM specific build flags are prefixed by ``NEOFOAM_``.

.. note::

   NeoFOAM will automatically enable ``Kokkos_ENABLE_CUDA`` or ``Kokkos_ENABLE_HIP`` if either of this is available on
   the system. This can be prevented by setting both options explicitly.

Building with CMake Presets
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additionally, we provide several CMake presets to set commonly required flags if you compile NeoFoam in combination with Kokkos.

   .. code-block:: bash

    cmake --list-presets # To list existing presets

To build NeoFOAM for production use, you can use the following commands:

   .. code-block:: bash

    cmake --preset production # To configure with ninja and common kokkos flags
    cmake --build --preset production # To compile with ninja and common kokkos flags

It should be noted that the build directory changes depending on the chosen preset. This way you can have different build directories for different presets and easily switch between them.

Building with Spack
^^^^^^^^^^^^^^^^^^^

A good way to simplify the process of building NeoFOAM is by using spack.
Here is a short tutorial on how to build NeoFOAM with spack for development.
First clone spack from  https://github.com/spack/spack.

   .. code-block:: bash

    git clone  https://github.com/spack/spack
    source spack/share/spack/setup-env.sh

Next we create a development environment for NeoFOAM and add NeoFOAM to it.

   .. code-block:: bash

    mkdir neofoam-env
    spack env create  -d neofoam-env
    spack env activate neofoam-env
    cd neofoam-env
    spack develop --path /home/greole/data/code/NeoFOAM neofoam

Next we install clang 17 as a compiler into our environment

   .. code-block:: bash

    spack add llvm@17
    spack compiler add "$(spack location -i llvm)"


Here is the current package.py. Edit spack/var/spack/repos/builtin/packages/neofoam/package.py accordingly

   .. code-block:: bash
        class Neofoam(CMakePackage):
            """NeoFOAM is a WIP prototype of a modern CFD core."""

            homepage = "https://github.com/exasim-project/NeoFOAM"
            git = "https://github.com/exasim-project/NeoFOAM.git"

            maintainers("greole", "HenningScheufler")

            license("MIT", checked_by="greole")
            version("main", branch="main")

            variant("cuda", default=False, description="Compile with CUDA support")
            variant("hip", default=False, description="Compile with HIP support")
            variant("ginkgo", default=True, description="Compile with Ginkgo")
            variant("sundials", default=True, description="Compile with Sundials")
            variant("test", default=False, description="Compile and install tutorial programs")

            depends_on("c", type="build")
            depends_on("cxx", type="build")
            depends_on("cmake@3.26:", type="build")
            depends_on("mpi@3")
            depends_on("cuda@12.6", when="+cuda")
            depends_on("kokkos@4.3.0")
            depends_on("ginkgo", when="+ginkgo")

            def cmake_args(self):
                return [
                    '-DNEOFOAM_BUILD_TESTS=%s' % ('+test' in self.spec),
                    '-DKokkos_ENABLE_CUDA=%s' % ('+cuda' in self.spec),
                ]


Next, we add NeoFOAM with the required dependencies.

   .. code-block:: bash

     spack add neofoam+test++cuda ^kokkos cuda_arch=80 cxxstd=20  ^ginkgo cuda_arch=80   %llvm@17
     spack install


Prerequisites
^^^^^^^^^^^^^

The following tools are used in the development of this project:

required tools for documentation:

.. code-block:: bash

    sudo apt install doxygen
    pip install pre-commit sphinx furo breathe sphinx-sitemap


required tools for compilation (ubuntu latest 24.04):

.. code-block:: bash

    sudo apt update
    sudo apt install \
    ninja-build \
    clang-16 \
    gcc-10 \
    libomp-16-dev \
    python3 \
    python3-dev \
    build-essential


Workflow with vscode
^^^^^^^^^^^^^^^^^^^^

install the following extensions:

.. code-block:: bash

   ms-vscode.cpptools
   ms-vscode.cmake-tools


After installation, you can open the NeoFOAM directory with vscode and configure the build with cmake presets with the cmake extension as shown below:

.. figure:: _static/installation/cmakePresets.gif
   :alt: configure the build with cmake presets
   :align: center

After configuring the build, you can build the project with the build button or test in "testing" tab (flask icon).

To create the documentation, you can use the 'Build Sphinx Documentation' task in the vscode task menu. Type `Ctrl+P` and type `task` and press space and the build documentation and press enter. The documentation will be created in the `docs_build` directory.
