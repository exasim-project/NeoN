# Conda recipe for NeoN

This recipe packages the NeoN Python bindings and ships the compiled extension
(`neon/_neon*.so` on Linux/macOS, `neon/_neon*.pyd` on Windows).

## Build locally

```bash
conda build conda/recipe
```

The recipe defaults to `CMAKE_BUILD_PARALLEL_LEVEL=2` to reduce memory pressure.
Override it when needed:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=1 conda build conda/recipe
```

Get the produced package path:

```bash
conda build conda/recipe --output
```

## Upload to anaconda.org

```bash
anaconda login
anaconda upload "$(conda build conda/recipe --output)"
```

## Install and verify

```bash
conda install -c <your-channel> neon
python -c "import neon; import neon._neon; print(neon.__version__)"
```

## Notes

- The package build uses `pip install . --no-build-isolation --no-deps` with `scikit-build-core`.
- Build requirements use Conda compilers to produce relocatable binaries.
- Additional system/compiler requirements may be needed depending on your platform and accelerator setup.
