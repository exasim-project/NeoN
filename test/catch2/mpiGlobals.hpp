// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <mpi.h>

// Define MPI comm as global variable, since the Catch reporter can't be constructed
// with an MPI comm. Thus, to still access it, it's stored globally
extern MPI_Comm COMM;

extern int ROOT;
extern int RANK;
extern int COMM_SIZE;

extern bool IS_ROOT;

// True when the MPI library actually provides MPI_THREAD_MULTIPLE, so the background IO
// serialization thread may call MPI concurrently with the main thread. When false the thread is
// never started and the reporter skips the serialization handshake entirely -- both sides must
// agree, or a failing assertion would block forever waiting for a peer that does not exist.
// Set once in test_main_mpi.cpp before Catch2 runs.
extern bool IO_SERIALIZATION;
