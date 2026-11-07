Contributing
^^^^^^^^^^^^

Contributions are highly welcome. Here is some information getting you started.

All participants are expected to follow the `NeoN Code of Conduct <https://github.com/exasim-project/NeoN/blob/develop/CODE_OF_CONDUCT.md>`_.

Reporting Issues and Getting Help
"""""""""""""""""""""""""""""""""

**Reporting bugs.**
If you encounter a bug, a build failure, or incorrect results, please open an issue on the
`GitHub issue tracker <https://github.com/exasim-project/NeoN/issues>`_.
Before opening a new issue, search the existing issues to check whether the problem has already been reported.
To help us reproduce the problem, please include the following where applicable:

 * **Version information:** NeoN version or commit hash
 * **Installation details:** installation method (source build, ``pip``, conda/pixi, or Spack) and Python version (if applicable)
 * **Environment:** operating system and, for source builds, compiler and CMake version
 * **Build configuration:** CMake preset or configuration options used (e.g., ``-DNeoN_WITH_MPI=OFF``) for source builds
 * **Compute platform:** Kokkos backend and hardware (e.g., CUDA on NVIDIA H100, HIP on AMD MI210, OpenMP on CPU), including GPU driver and toolkit version if applicable
 * **Reproducible example:** a minimal, self-contained example that reproduces the problem
 * **Error details:** the full error message or log output
 * **Expected vs. observed behaviour:** what you expected to happen and what happened instead

**Requesting features.**
Feature requests and suggestions for improvements are also welcome as GitHub issues.
Please describe the intended use case and, if possible, sketch how the feature could look from a user's perspective.

**Asking questions and seeking help.**
For questions about using NeoN, integrating it into another CFD framework, or where to start contributing,
please use `GitHub Discussions <https://github.com/exasim-project/NeoN/discussions>`_
or join the community chat on `Gitter <https://matrix.to/#/#NeoFOAM:gitter.im>`_.
Questions asked in public help other users as well, so we prefer these channels over private messages.

NeoN Code Style Guide
""""""""""""""""""""""""

To simplify coding and code reviews you should adhere to the following code style. However, most
of NeoNs code style guide is checked via automated tools.
For example, formatting and non-format related code style is enforced via clang-format and clang-tidy.
Corresponding configuration files are `.clang-format <https://github.com/exasim-project/NeoN/blob/develop/.clang-format>`_
and `.clang-tidy <https://github.com/exasim-project/NeoN/blob/develop/.clang-tidy>`_.
Furthermore, adequate licensing of the source code is checked via the reuse linter and typos checks for obvious spelling issues.
Thus, this style guide doesn't list all stylistic rules explicitly but rather gives advice for ambiguous situations and mentions the rational for some decisions.

 * We generally try to comply with the `C++ core guidelines <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines>`_.
 * Use `camelCase` for functions and members and capitalized `CamelCase` for classes
 * Use descriptive template parameter names.
   For example, prefer ``template <class ValueType>`` over ``template <class T>`` if the type can be a float or double.
 * Use ``[[nodiscard]]`` except for getters.
   To indicate that a function simply returns a value instead of performing expensive computations, ``[[nodiscard]]`` can be omitted.
 * Use ``final`` except for abstract classes.
   NeoNs aims at having a rather flat inheritance hierarchy and composition is generally preferred.
   To avoid unintended inheritance, using `final` is advised.
 * Avoid storing references as data members in classes.
   An exception might be a ``const Unstructured& mesh`` as it will most likely outlive any other objects.
 * Private member variables should be suffixed with ``_``.
 * Order of parameters to functions should be in, in-out, out.
   E.g. ``foo(const Type& in, Type& inOut, Type& out)`` instead of ``foo(Type& inOut,  Type& out, const Type& in)`` or some other variation.
 * Use US spelling

Folder structure and file names:

 * Use `camelCase` for files and folder names
 * The location of implementation and header files should be consistent.
   i.e. a file ``src/model/a.cpp`` should have a header file in ``include/NeoN/model/a.hpp`` and the corresponding test implementation in ``test/model/a.cpp``
 * File and folder names should not be redundant. Ie. ``finiteVolume/cellCentred/geometryModel/finiteVolumecellCentredgeometryModel.hpp`` should be
   ``finiteVolume/cellCentred/geometryModel/model.hpp``.

Collaboration via Pull Requests
"""""""""""""""""""""""""""""""

If you want to contribute a specific feature or fix, please don't hesitate to open a PR. After creating the PR, the following process is applied.

 * Request a review by a given person or set the Ready-for-Review label.
 * At least one (ideally two) approval(s) is/are required before a PR can be merged.
 * Make sure that all required pipelines succeed.
 * If you add new features, make sure to provide sufficient unit tests.
   Also at some point before merging you can add the ``full-ci`` label to include build and tests on GPU hardware.
 * If you add a new feature or bug-fix, please add an entry to the ``CHANGELOG.md`` file.
   For pure refactor PRs, the ``skip-changelog`` label can be set.
 * If the PR should be merged in a specific order or if you don't have the permission to merge, add the ``ready-to-merge`` label.
 * Before merging, make sure to rebase your PR on to the latest state of ``develop``.
 * Make sure to add yourself to the ``AUTHORS.md`` file.
 * To simplify the review process, small to medium sized PRs are preferred.
   If your PR exceeds 1000 lines of code changes, consider to break the PR up into smaller PRs which can be merged via `stacked PRs <https://graphite.dev/guides/stacked-diffs>`_.

Github Workflows and Labels
"""""""""""""""""""""""""""

In order to influence the workflow pipeline and to structure the review process, several labels exist.
The following labels and their meaning are discussed here:

 * ``ready-for-review``: Signal that the PR is ready for review.
 * ``ready-to-merge``: Signal that the work on this PR has been finished and can be merged.
 * ``skip-changelog``: Don't check whether the changelog has been updated. Use this label if the changes are not any new features or bug-fixes.
 * ``skip-build``: Don't run any build steps, including building the compile commands database.
 * ``skip-cache``: Don't cache the build folders. Forces to rebuild the build folder after every push to GitHub.
 * ``full-ci``: Run tests on AWS and enable sanitizer workflows.
 * ``auto-fix``: Apply changes from pre-commit and clang-tidy.
 * ``benchmark``: Run the benchmark jobs on GPUs in the LRZ pipeline.
 * ``skip-nvidia``, ``skip-amd``, ``skip-intel``: Skip the GPU tests on the respective vendor's hardware in the LRZ pipeline.

A full list of the labels can be found `here <https://github.com/exasim-project/NeoN/labels>`_.

Building the Documentation
""""""""""""""""""""""""""

NeoN's documentation can be found `main <https://exasim-project.com/NeoN/latest/index.html>`_  and `doxygen <https://exasim-project.com/NeoN/latest/doxygen/html/>`_ documentation can be found online. However, if you want to build the documentation locally, you can do so by executing the following steps.
First, make sure that Sphinx and Doxygen are installed on your system. Second, execute the following commands:

   .. code-block:: bash

    cmake -B build -DNeoN_BUILD_DOC=ON # configure the build
    cmake --build build --target sphinx # build the documentation
    # or
    sphinx-build -b html ./doc ./docs_build


The documentation will be built in the ``docs_build`` directory and can be viewed by opening the ``index.html`` file in a web browser.

   .. code-block:: bash

    firefox docs_build/index.html

Alternatively, the documentation can be built by just adding the ``-DNeoN_BUILD_DOC=ON`` to the configuration step of the build process and then building the documentation using the ``sphinx`` target.
