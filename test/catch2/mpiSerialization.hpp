// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#pragma once

#include <mpi.h>

constexpr int SERIALIZATION_TAG = 6;

// Function to serialize IO, has to be run in a separate thread.
// To initiate writing to stdout or stderr a thread has to send a single bool to the ROOT
// process with the tag 666, and then receive a single bool back. After that, the process
// can write out. After writing is done the process has to send again a bool back to ROOT.
// The order in which processes are allowed to write is not deterministic.
void serializeIO(volatile bool* threadShutdown);
