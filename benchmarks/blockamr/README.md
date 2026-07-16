<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
-->

# blockAMR backend benchmark (`jax` vs `cpp`)

Compares the two explicit-path backends of the blockAMR `Equation` API on the
**same** equation, with discretisation (`fvSchemes`) and solution approach
(`fvSolution`, incl. the backend) read from disk. Parameterised over cell count
and max grid size; scheme is set in the case file.

```bash
python bench_backends.py --n-cell 32 64 --max-size 16 32 64 --steps 100
```

## Harness flow

```mermaid
flowchart TD
    Y1["system/fvSchemes.yaml"] --> R1["read_fv_schemes()<br/>pydantic FvSchemes"]
    Y2["system/fvSolution.yaml"] --> R2["read_fv_solution()<br/>pydantic FvSolution"]
    R1 --> S["schemes dict<br/>{'div(phi,U)':'vanLeer', …}"]
    R2 --> SOL["solution dict<br/>{'backend': …}"]

    S --> EQ["Equation(exp.ddt(U) + exp.div(phi,U)<br/>- exp.laplacian(nu,U), schemes=…)<br/><b>built once</b>"]

    EQ --> LOOP{for backend in jax, cpp}
    SOL -->|"override backend key"| LOOP
    LOOP --> W["1 solve = compile/build<br/>+ warmup steps"]
    W --> T["timed loop: N × UEqn.solve(dt, t, solution)<br/>sync after each step"]
    T --> SNAP["snapshot U"]
    SNAP --> P["parity: allclose(jax, cpp)"]
    P --> REP["table: compile_ms · per_step_ms · Mcell/s"]
```

## Data loading (pydantic over YAML)

```mermaid
flowchart LR
    subgraph disk["case/system/*.yaml"]
        A["fvSchemes.yaml<br/>div(phi,U): vanLeer"]
        B["fvSolution.yaml<br/>solvers.U.backend: jax|cpp"]
    end
    subgraph models["pydantic schema = the validator"]
        M1["FvSchemes<br/>RootModel[dict[str,str]]"]
        M2["FvSolution → SolverBlock<br/>backend: Literal['jax','cpp']"]
    end
    A --> M1 --> D1["schemes: dict → Equation(schemes=)"]
    B --> M2 --> D2["solution: dict → .solve(solution=)"]
    M2 -. "bad backend name<br/>fails at load" .-> ERR["ValidationError"]
```

## Backend dispatch — where the cost is

```mermaid
flowchart TD
    SV["UEqn.solve()"] --> FS["free solve()<br/>resolve schemes by name"]
    FS --> ES["backend.euler_step(equation, U, lev, t, dt)"]

    ES -->|"backend = jax"| JX
    ES -->|"backend = cpp"| CP

    subgraph JX["JaxBackend.parallel_for — per step, HOST-side"]
        J1["contiguous_array() · fab_metadata()"]
        J2["build_tile_layout · tree.flatten(kernel)"]
        J3["Pallas dispatch ×ncomp (3 for U)"]
        J4["per-box loop: _gather_valid + jnp.stack"]
        J1 --> J2 --> J3 --> J4
    end

    subgraph CP["CppBackend — on DEVICE"]
        C1["one AMReX ParallelFor<br/>accumulate terms → Euler update"]
    end

    J4 -.->|"scales with #boxes × ncomp"| SLOW["≈ 7–235× slower"]
    C1 -.->|"box-count invariant"| FAST["≈ 1 ms/step"]
```

## GPU memory split (shared device)

```mermaid
flowchart LR
    GPU["GPU 12.3 GB"] --> JAXM["jax preallocated<br/>XLA_PYTHON_CLIENT_MEM_FRACTION=0.35<br/>≈ 4.3 GB"]
    GPU --> AMR["AMReX arena<br/>AMREX_THE_ARENA_INIT_SIZE=0<br/>grows on demand"]
    JAXM --> OK["jax + AMReX < 100%<br/>(no OOM, no starvation)"]
    AMR --> OK
```

## Findings (RTX-class 12 GB, `vanLeer` div + central laplacian, ncomp=3)

| n_cell | max_size | boxes | jax ms/step | cpp ms/step | jax slowdown |
|-------:|---------:|------:|------------:|------------:|-------------:|
| 64 | 16 | 64 | 254.9 | 1.08 | 236× |
| 64 | 32 |  8 |  37.8 | 0.87 |  43× |
| 64 | 64 |  1 |   8.9 | 1.21 |   7× |

jax per-step scales with **box count** (host-side per-box Python in
`parallel_for`); cpp is box-count invariant (one on-device `ParallelFor`). Parity
holds (`allclose`, jax≈cpp) on every row. Improving jax means batching the
`ncomp` dispatches and moving the per-box gather/stack onto the device.
