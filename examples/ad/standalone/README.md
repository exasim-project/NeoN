# Standalone forward-mode AD example

Builds with a plain host compiler; no Kokkos, CMake or MPI required.

```
g++ -std=c++20 -Ishim -I../../../include forwardSensitivity.cpp -o forwardSensitivity
./forwardSensitivity
```

Exit status is 0 if every AD gradient matches its central-difference reference.

`shim/Kokkos_Core.hpp` is a minimal stand-in used only by the command above, so
that the AD primitives can be verified without a full NeoN build. A normal NeoN
build resolves `<Kokkos_Core.hpp>` to the real header.

