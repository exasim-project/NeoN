Installation
============

You can build NeoN by following these steps:

Clone the NeoN repository:

   .. code-block:: bash

      git clone https://github.com/exasim-project/NeoN.git

Navigate to the NeoN directory:

   .. code-block:: bash

      cd NeoN

NeoN uses CMake to build, thus the standard CMake procedure should work, however, we recommend using one of the provided CMake presets detailed below in :ref:`Building with CMake Presets`. From a build directory, you can execute:

   .. code-block:: bash

        mkdir build
        cd build
        cmake <DesiredBuildFlags> ..
        cmake --build . -j<the number of CPU cores>
        cmake --install .

By default, the command `cmake --build . -j` will use all the CPU cores available. It can consume a significant
amount of memory when building with many cores. If you want to limit the number of cores used during the build step,
you can specify the number of CPU cores as shown above.

The following can be chained with -D<DesiredBuildFlags>=<Value> to the CMake command.
Most relevant build flags are:

+------------------+--------------------------------+---------+
| Flag             | Description                    | Default |
+==================+================================+=========+
| CMAKE_BUILD_TYPE | Build in debug or release mode | Debug   |
+------------------+--------------------------------+---------+
| NeoN_BUILD_DOC   | Build NeoN with documentation  | ON      |
+------------------+--------------------------------+---------+
| NeoN_BUILD_TESTS | Build NeoN with tests          | OFF     |
+------------------+--------------------------------+---------+

To browse the full list of build options it is recommended to use a build tool like ``ccmake``.
By opening the the project with cmake-gui you can easily set these flags and configure the build.
NeoN specific build flags are prefixed by ``NeoN_``.

Building for GPUs
^^^^^^^^^^^^^^^^^^
NeoN will automatically enable ``Kokkos_ENABLE_CUDA`` or ``Kokkos_ENABLE_HIP`` if either of this is available on
the system. This can be prevented by setting both options explicitly.

If NeoN does not detect the GPU backend automatically, you can set some relevant flags to enable GPU support
during the configure step.

For NVIDIA GPUs, specifying the GPU architecture via ``CMAKE_CUDA_ARCHITECTURES`` should be sufficient.

.. code-block:: bash

   -DCMAKE_CUDA_ARCHITECTURES=<GPU_ARCH>

For AMD GPUs, you may need to set up some relevant HIP environment variables before the configure step.

.. code-block:: bash

   export PATH=/opt/rocm/bin:$PATH
   export HIPCC_CXX=/usr/bin/g++  # If you want to use g++ as the host compiler

Then you can enable HIP during the configure step with the following flags.

.. code-block:: bash

   -DCMAKE_CXX_COMPILER=hipcc
   -DCMAKE_HIP_ARCHITECTURES=<GPU_ARCH>
   -DKokkos_ARCH_AMD_<GPU_ARCH>=ON  # e.g., -DKokkos_ARCH_AMD_GFX90A=ON

In the case of Intel PVC GPUs, the following flags ensure sycl support.

.. code-block:: bash

   -DCMAKE_CXX_COMPILER=icpx
   -DCMAKE_CXX_FLAGS=-fsycl
   -DCMAKE_BUILD_TYPE=release
   -DKokkos_ENABLE_SYCL=ON
   -DKokkos_ARCH_INTEL_PVC=ON

Please note that current support for NeoN on Intel GPUs is experimental.

After configuring for GPU support, you can continue to build NeoN.

.. _Building with CMake Presets:

Building with CMake Presets
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additionally, we provide several CMake presets to set commonly required flags if you compile NeoN in combination with Kokkos.

   .. code-block:: bash

    cmake --list-presets # To list existing presets

To build NeoN for production use, you can use the following commands:

   .. code-block:: bash

    cmake --preset production # To configure with ninja and common kokkos flags
    cmake --build --preset production # To compile with ninja and common kokkos flags

It should be noted that the build directory changes depending on the chosen preset. This way you can have different build directories for different presets and easily switch between them.

Building with Spack
^^^^^^^^^^^^^^^^^^^

A good way to simplify the process of building NeoN is by using spack.
Here is a short tutorial on how to build NeoN with spack for development.
First clone spack from  https://github.com/exasim-project/spack (until neon is fully merged into spack).

   .. code-block:: bash

    git clone https://github.com/exasim-project/spack -b neofoam
    source spack/share/spack/setup-env.sh

Next we create a development environment for NeoN and add NeoN to it.

   .. code-block:: bash

    mkdir NeoN-env
    spack env create  -d NeoN-env
    spack env activate NeoN-env
    cd NeoN-env
    spack develop --path /home/greole/data/code/NeoN neon

Next we install clang 17 as a compiler into our environment

   .. code-block:: bash

    spack add llvm@17
    spack install
    spack compiler add "$(spack location -i llvm)"

Next, we add NeoN with the required dependencies.

   .. code-block:: bash

     spack add neon+test++cuda ^kokkos cuda_arch=80 cxxstd=20  ^ginkgo cuda_arch=80   %llvm@17
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


After installation, you can open the NeoN directory with vscode and configure the build with cmake presets with the cmake extension as shown below:

.. figure:: _static/installation/cmakePresets.gif
   :alt: configure the build with cmake presets
   :align: center

After configuring the build, you can build the project with the build button or test in "testing" tab (flask icon).

To create the documentation, you can use the 'Build Sphinx Documentation' task in the vscode task menu. Type `Ctrl+P` and type `task` and press space and the build documentation and press enter. The documentation will be created in the `docs_build` directory.
