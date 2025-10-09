Overview
========
The **NeoN** project uses a two-level Continuous Integration (CI) system
to ensure correct builds, GPU compatibility, and automated benchmarking.

The main repository is hosted on **GitHub**, and GPU-based workflows are delegated
to **LRZ GitLab**, where jobs are executed on both **NVIDIA** and **AMD** GPUs.

.. figure:: _static/ci_neon_architecture.svg
   :align: center
   :alt: Overview of the CI architecture for NeoN
   :figwidth: 90%

   *Figure 1 – Overview of NeoN’s CI architecture integrating GitHub and LRZ GitLab.*

.. contents::
   :local:
   :depth: 2

-------------------------------
Continuous Integration on GitHub
-------------------------------
GitHub CI is responsible for managing the overall NeoN CI workflow.

**Responsibilities:**

* Build and test NeoN on **CPU** platforms.
* Push the source code and commit metadata to **LRZ GitLab**.
* Cancel outdated pipelines on LRZ GitLab for the same branch.
* Trigger new LRZ GitLab pipelines for GPU builds and benchmarks.

.. note::
   The GitHub CI acts as the *control layer* for all NeoN CI operations.
   Developers interact only with GitHub — all LRZ GitLab pipelines are triggered automatically.

-------------------------------
Continuous Integration on LRZ GitLab
-------------------------------
The LRZ GitLab CI handles GPU-related operations.

**Responsibilities:**

* Build and test NeoN on **NVIDIA** and **AMD** GPU nodes.
* Run benchmark jobs after successful build and test stages.
* Report the status and results back to GitHub for unified monitoring.

.. figure:: _static/ci_neon_layers.svg
   :align: center
   :alt: Two-layer CI structure for NeoN
   :figwidth: 90%

   *Figure 2 – Two-layer structure of the NeoN CI pipeline.*

.. _ci-neon-workflow:

Development Workflow
====================
The CI workflow for NeoN proceeds as follows:

#. A developer pushes a commit or pull request to GitHub.
#. GitHub CI builds and tests NeoN on CPUs.
#. GitHub CI pushes the same branch to LRZ GitLab.
#. Existing LRZ GitLab pipelines for that branch are canceled.
#. GitHub CI triggers a **new LRZ GitLab pipeline**.
#. LRZ GitLab CI builds and tests NeoN on GPUs.
#. *(Optional)* Benchmark jobs are executed after successful testing.
#. The developer monitors all results directly on GitHub.

.. figure:: _static/ci_neon_workflow.svg
   :align: center
   :alt: Workflow diagram of NeoN CI
   :figwidth: 90%

   *Figure 3 – Step-by-step workflow of NeoN’s CI integration.*

.. _ci-neon-mechanism:

Detailed Mechanism
==================
The internal triggering and coordination logic is as follows:

1. GitHub CI starts the pipeline on **NeoN LRZ GitLab**.
2. The GitHub workflow checks whether the corresponding branch exists on LRZ GitLab.
3. If successful, the LRZ GitLab pipeline performs all GPU build-and-test jobs.
4. *(Optional)* Benchmarking is triggered on LRZ GitLab after successful validation.

**Branch Handling Rules:**

* If the branch exists on LRZ GitLab, it is used directly.
* Otherwise, the **main** branch is used as a fallback.

.. tip::
   Use the ``benchmark`` label on a NeoN pull request to trigger benchmarking jobs.

.. _ci-neon-labels:

Pull Request Labels
===================
NeoN’s GitHub repository uses labels to control the CI behavior.

**Available Labels:**

* ``Skip-build`` — Skip all build-and-test jobs on both GitHub and LRZ GitLab.
* ``benchmark`` — Enable GPU benchmarking jobs after successful builds.

These labels allow developers to fine-tune which stages of the CI system should run.

.. _ci-neon-summary:

Summary
=======
The NeoN CI system provides:

* Unified GitHub-driven CI management.
* Transparent CPU and GPU build workflows.
* Automatic synchronization between GitHub and LRZ GitLab.
* Branch-aware pipeline handling and cancellation.
* On-demand GPU benchmarking via PR labels.

.. seealso::

   * :ref:`ci-neon-workflow`
   * :ref:`ci-neon-mechanism`
   * :ref:`ci-neon-labels`
