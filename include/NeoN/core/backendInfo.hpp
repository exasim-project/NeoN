// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

namespace NeoN
{

[[nodiscard]] bool hasSerialBackend() noexcept;

[[nodiscard]] bool hasCpuBackend() noexcept;

[[nodiscard]] bool hasGpuBackend() noexcept;

} // namespace NeoN
