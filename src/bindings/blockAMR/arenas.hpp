// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>

#include <cstring>
#include <vector>

// Custom AMReX Arenas used by the NeoN Python bindings.
//
//  - ExternalArena: serves a fixed external pointer (e.g. JAX/numpy buffer)
//    on the first matching alloc; everything else falls through to a base
//    arena. Used to construct MultiFabs that view foreign memory zero-copy.
//
//  - PaddedArena: a sequential bump allocator that overallocates a single
//    chunk so contiguous_array() returns a stable shape across regrids
//    (avoids JAX recompilation). Combined with MFInfo::SetAllocSingleChunk,
//    it makes every MultiFab fab live in one packed buffer — including EB
//    MultiFabs whose factory honours the arena.

namespace neon::bindings {

class ExternalArena final : public amrex::Arena
{
    void* m_ptr;
    std::size_t m_size;
    amrex::Arena* m_fallback;
    bool m_chunk_served;
    bool m_device_accessible;
    bool m_host_accessible;

public:
    ExternalArena(void* p, std::size_t sz, amrex::Arena* fallback,
                  bool device, bool host)
        : m_ptr(p),
          m_size(sz),
          m_fallback(fallback),
          m_chunk_served(false),
          m_device_accessible(device),
          m_host_accessible(host)
    {}

    [[nodiscard]] void* alloc(std::size_t sz) override
    {
        if (!m_chunk_served && sz == m_size)
        {
            m_chunk_served = true;
            return m_ptr;
        }
        return m_fallback->alloc(sz);
    }

    void free(void* pt) override
    {
        if (pt == m_ptr) { return; }
        m_fallback->free(pt);
    }

    [[nodiscard]] bool isDeviceAccessible() const override { return m_device_accessible; }
    [[nodiscard]] bool isHostAccessible() const override { return m_host_accessible; }
    [[nodiscard]] bool isManaged() const override
    {
        return m_device_accessible && m_host_accessible;
    }
    [[nodiscard]] bool isDevice() const override
    {
        return m_device_accessible && !m_host_accessible;
    }
    [[nodiscard]] bool isPinned() const override
    {
        return m_host_accessible && m_device_accessible;
    }
};

class PaddedArena final : public amrex::Arena
{
    void* m_buf;
    std::size_t m_valid_size;
    std::size_t m_padded_size;
    amrex::Arena* m_base;
    bool m_chunk_served;
    bool m_device_accessible;
    bool m_host_accessible;

public:
    PaddedArena(amrex::Arena* base, std::size_t valid_bytes,
                std::size_t padded_bytes, bool device, bool host)
        : m_buf(nullptr),
          m_valid_size(valid_bytes),
          m_padded_size(padded_bytes),
          m_base(base),
          m_chunk_served(false),
          m_device_accessible(device),
          m_host_accessible(host)
    {
        m_buf = base->alloc(padded_bytes);
        if (padded_bytes > valid_bytes)
        {
            char* pad_start = static_cast<char*>(m_buf) + valid_bytes;
            std::size_t pad_n = padded_bytes - valid_bytes;
            if (device && !host)
            {
                std::vector<char> zeros(pad_n, 0);
                amrex::Gpu::htod_memcpy(pad_start, zeros.data(), pad_n);
            }
            else
            {
                std::memset(pad_start, 0, pad_n);
            }
        }
    }

    ~PaddedArena() override
    {
        if (m_buf) { m_base->free(m_buf); }
    }

    [[nodiscard]] void* alloc(std::size_t sz) override
    {
        if (!m_chunk_served && sz == m_valid_size)
        {
            m_chunk_served = true;
            return m_buf;
        }
        return m_base->alloc(sz);
    }

    void free(void* pt) override
    {
        if (pt == m_buf) { return; }
        m_base->free(pt);
    }

    [[nodiscard]] std::size_t paddedSize() const { return m_padded_size; }

    [[nodiscard]] bool isDeviceAccessible() const override { return m_device_accessible; }
    [[nodiscard]] bool isHostAccessible() const override { return m_host_accessible; }
    [[nodiscard]] bool isManaged() const override
    {
        return m_device_accessible && m_host_accessible;
    }
    [[nodiscard]] bool isDevice() const override
    {
        return m_device_accessible && !m_host_accessible;
    }
    [[nodiscard]] bool isPinned() const override
    {
        return m_host_accessible && m_device_accessible;
    }
};

inline amrex::Arena* pickArena(const std::string& memory)
{
    using namespace amrex;
    if (memory == "device")  return The_Device_Arena();
    if (memory == "managed") return The_Managed_Arena();
    if (memory == "pinned")  return The_Pinned_Arena();
    return The_Arena();
}

} // namespace neon::bindings
