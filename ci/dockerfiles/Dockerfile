# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#
# NeoN CI image: a Ginkgo image of exasim-project/ginkgo-packaging plus what the
# scripts in ci/lrz-gitlab need. The Ginkgo image brings the GPU toolchain, MPI
# (GPU-aware MPICH in /opt/mpich for cuda and rocm, Intel MPI for sycl) and Ginkgo
# in /opt/ginkgo. GINKGO_TAG is set in .github/workflows/ci-images.yaml:
#   docker build -f ci/dockerfiles/Dockerfile \
#     --build-arg GINKGO_BACKEND=cuda --build-arg GINKGO_TAG=<tag> ci/dockerfiles
# GINKGO_BACKEND is cuda (NVIDIA), rocm (AMD) or sycl (Intel).

ARG GINKGO_BACKEND
ARG GINKGO_TAG
FROM ghcr.io/exasim-project/ginkgo-${GINKGO_BACKEND}:${GINKGO_TAG}

ARG GINKGO_BACKEND
ARG GINKGO_TAG
ARG DEBIAN_FRONTEND=noninteractive

# pre-commit: the develop preset sets NeoN_DEVEL_TOOLS=ON, which requires it
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pre-commit \
    python3-pip \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# scripts/catch2json.py, run by ci/lrz-gitlab/benchmark.sh
RUN pip3 install --no-cache-dir --break-system-packages xmltodict

LABEL org.opencontainers.image.source="https://github.com/exasim-project/NeoN" \
      org.opencontainers.image.description="NeoN CI image: Ginkgo ${GINKGO_TAG} (${GINKGO_BACKEND})"
