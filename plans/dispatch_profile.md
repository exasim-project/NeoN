# Dispatch Performance Profile — 128³ Laplacian

## Component Breakdown

| Component | Time (ms) | Notes |
|-----------|----------|-------|
| `contiguous_array()` | 0.011 | Zero-copy pointer to GPU buffer |
| `build_tile_layout()` (C++) | 0.013 | MFIter + htod_memcpy |
| `jax.tree.flatten(kernel)` | 0.001 | Walk equinox tree |
| `_run_pallas` (JIT dispatch + kernel) | 0.109 | JAX JIT lookup + Pallas kernel |
| `copy_from_flat()` | 0.032 | Single dtod_memcpy |
| **parallel_for total** | **0.166** | **Sum of above** |
| **C++ laplacian** | **0.129** | **AMReX ParallelFor** |
| **Ratio (parallel_for / C++)** | **1.3x** | |

## DSL evaluate() overhead

| Component | Time (ms) | Notes |
|-----------|----------|-------|
| `parallel_for` (above) | 0.166 | The actual dispatch |
| `build_kernel_3d` + `CombinedSource` | ~0.17 | Equinox module construction |
| Temp MultiFab + `set_val(0.0)` | ~0.15 | Allocation for evaluate output |
| `out_mf.arrays()` + per-box slicing | ~0.10 | Extract valid results |
| `fill_patch` | ~0.05 | Ghost cell exchange |
| Python overhead | ~0.06 | Function calls, attribute lookups |
| **DSL evaluate() total** | **0.696** | |
| **Ratio (evaluate / C++)** | **5.4x** | |

## Key finding

`parallel_for` alone is **1.3x C++**. The 5.4x gap in `evaluate()` is
from the DSL wrapper (kernel construction, temp MultiFab, result extraction).

For `solve()` (writes in-place, no temp MultiFab, no result extraction):
```
parallel_for:    0.166 ms
fill_patch:      0.05 ms
build_kernel_3d: 0.17 ms
Total solve:     ~0.39 ms → 3x C++
```

## Where the time goes in `_run_pallas` (0.109ms)

The 0.109ms for `_run_pallas` includes:
- JAX JIT cache lookup: ~0.02ms (hash static args, find compiled fn)
- JAX argument staging: ~0.02ms (wrap arrays for XLA)
- GPU kernel launch: ~0.03ms (CUDA launch overhead)
- GPU kernel compute: ~0.04ms (actual Triton stencil)

The `static_argnums` vs closure approach shows no difference (0.109 vs 0.113ms).
The overhead is in JAX's dispatch path, not in our hashing.

## Comparison: C++ breakdown

The C++ `laplacian` at 0.129ms includes:
- MFIter loop (1 box): ~0.001ms
- ParallelFor GPU launch: ~0.03ms
- GPU kernel compute: ~0.10ms (no Triton fusion, separate kernel)

The Pallas kernel compute (0.04ms) is faster than C++ kernel compute (0.10ms)
because Triton optimizes memory access patterns. But C++ has less launch
overhead (direct CUDA launch vs JAX dispatch).

## Opportunities to close the gap

| Optimization | Saves | Feasibility |
|-------------|-------|-------------|
| Avoid kernel rebuild in `solve()` | 0.17ms | Medium — cache kernel if expression unchanged |
| Avoid temp MultiFab in `evaluate()` | 0.15ms | Easy — return flat buffer, extract lazily |
| Avoid per-box slicing in `evaluate()` | 0.10ms | Easy — return flat buffer directly |
| Reduce JAX dispatch overhead | 0.07ms | Hard — JAX framework cost |
| **Total recoverable** | **0.49ms** | |

With all optimizations, `evaluate()` would be ~0.21ms (1.6x C++).
For `solve()`, it would be ~0.22ms (1.7x C++).

At 256³, the kernel compute dominates and both converge to ~1x.
