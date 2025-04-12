# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from spack.package import *

class NeoN(CMakePackage):
    """NeoN is a WIP prototype of a modern CFD core."""

    homepage = "https://github.com/exasim-project/NeoN"
    git = homepage

    # maintainers("github_user1", "github_user2")

    license("UNKNOWN", checked_by="github_user1")

    version("develop", branch="main")

    depends_on("mpi")
    depends_on("kokkos@4.3.0")
