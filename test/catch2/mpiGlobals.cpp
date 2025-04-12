// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#include "mpiGlobals.hpp"

#include <chrono>
#include <thread>
#include <vector>


// Define MPI comm as global variable, since the Catch reporter can't be constructed
// with an MPI comm. Thus, to still access it, it's stored globally
MPI_Comm COMM = MPI_COMM_WORLD;

int ROOT = 0;
int RANK;
int COMM_SIZE;

bool IS_ROOT = false;
