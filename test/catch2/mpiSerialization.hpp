// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>

#include <mpi.h>

constexpr int SERIALIZATION_TAG = 6;

// Function to serialize IO, has to be run in a separate thread.
// To initiate writing to stdout or stderr a thread has to send a single bool to the ROOT
// process with the tag SERIALIZATION_TAG, and then receive a single bool back. After that, the
// process can write out. After writing is done the process has to send again a bool back to ROOT.
// The order in which processes are allowed to write is not deterministic.
//
// This calls MPI concurrently with the main thread, so it may only be started when the MPI library
// provides MPI_THREAD_MULTIPLE -- see IO_SERIALIZATION in mpiGlobals.hpp. threadShutdown is atomic
// rather than volatile: volatile orders nothing between threads and is not a synchronisation
// primitive.
void serializeIO(std::atomic<bool>* threadShutdown);
