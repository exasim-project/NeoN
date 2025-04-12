// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#include <mpi.h>

// Define MPI comm as global variable, since the Catch reporter can't be constructed
// with an MPI comm. Thus, to still access it, it's stored globally
extern MPI_Comm COMM;

extern int ROOT;
extern int RANK;
extern int COMM_SIZE;

extern bool IS_ROOT;
