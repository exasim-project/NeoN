// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/finiteVolume/cellCentred/operators/tensorOps.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred
{

VolumeField<SymmTensor> symm(const VolumeField<Tensor>& T)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<SymmTensor>>(T.mesh());
    VolumeField<SymmTensor> result(T.exec(), "symm", T.mesh(), bcs);

    auto inView = T.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        T.exec(),
        {0, T.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::symm(inView[i]); },
        "symmField"
    );
    return result;
}

VolumeField<Tensor> skew(const VolumeField<Tensor>& T)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<Tensor>>(T.mesh());
    VolumeField<Tensor> result(T.exec(), "skew", T.mesh(), bcs);

    auto inView = T.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        T.exec(),
        {0, T.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::skew(inView[i]); },
        "skewField"
    );
    return result;
}

VolumeField<scalar> mag(const VolumeField<Tensor>& T)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(T.mesh());
    VolumeField<scalar> result(T.exec(), "mag", T.mesh(), bcs);

    auto inView = T.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        T.exec(),
        {0, T.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::mag(inView[i]); },
        "magTensorField"
    );
    return result;
}

VolumeField<scalar> mag(const VolumeField<SymmTensor>& S)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(S.mesh());
    VolumeField<scalar> result(S.exec(), "mag", S.mesh(), bcs);

    auto inView = S.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        S.exec(),
        {0, S.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::mag(inView[i]); },
        "magSymmTensorField"
    );
    return result;
}

VolumeField<SymmTensor> dev(const VolumeField<SymmTensor>& S)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<SymmTensor>>(S.mesh());
    VolumeField<SymmTensor> result(S.exec(), "dev", S.mesh(), bcs);

    auto inView = S.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        S.exec(),
        {0, S.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::dev(inView[i]); },
        "devField"
    );
    return result;
}

VolumeField<SymmTensor> twoSymm(const VolumeField<Tensor>& T)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<SymmTensor>>(T.mesh());
    VolumeField<SymmTensor> result(T.exec(), "twoSymm", T.mesh(), bcs);

    auto inView = T.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        T.exec(),
        {0, T.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::twoSymm(inView[i]); },
        "twoSymmField"
    );
    return result;
}

} // namespace NeoN
