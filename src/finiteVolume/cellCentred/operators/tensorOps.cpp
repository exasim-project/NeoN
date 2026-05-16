// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <cmath>

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

VolumeField<scalar> mag(const VolumeField<Vec3>& v)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(v.mesh());
    VolumeField<scalar> result(v.exec(), "mag", v.mesh(), bcs);

    auto inView = v.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        v.exec(),
        {0, v.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::mag(inView[i]); },
        "magVec3Field"
    );
    return result;
}

VolumeField<scalar> inner(const VolumeField<Vec3>& v1, const VolumeField<Vec3>& v2)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(v1.mesh());
    VolumeField<scalar> result(v1.exec(), "inner", v1.mesh(), bcs);

    auto view1 = v1.internalVector().view();
    auto view2 = v2.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        v1.exec(),
        {0, v1.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = view1[i] & view2[i]; },
        "innerField"
    );
    return result;
}

VolumeField<scalar> max(const VolumeField<scalar>& f, scalar val)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(f.mesh());
    VolumeField<scalar> result(f.exec(), "max", f.mesh(), bcs);

    auto inView = f.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        f.exec(),
        {0, f.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = inView[i] > val ? inView[i] : val; },
        "maxField"
    );
    return result;
}

VolumeField<scalar> max(const VolumeField<scalar>& f1, const VolumeField<scalar>& f2)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(f1.mesh());
    VolumeField<scalar> result(f1.exec(), "max", f1.mesh(), bcs);

    auto view1 = f1.internalVector().view();
    auto view2 = f2.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        f1.exec(),
        {0, f1.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = view1[i] > view2[i] ? view1[i] : view2[i]; },
        "maxFieldField"
    );
    return result;
}

VolumeField<scalar> min(const VolumeField<scalar>& f, scalar val)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(f.mesh());
    VolumeField<scalar> result(f.exec(), "min", f.mesh(), bcs);

    auto inView = f.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        f.exec(),
        {0, f.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = inView[i] < val ? inView[i] : val; },
        "minField"
    );
    return result;
}

VolumeField<scalar> min(const VolumeField<scalar>& f1, const VolumeField<scalar>& f2)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(f1.mesh());
    VolumeField<scalar> result(f1.exec(), "min", f1.mesh(), bcs);

    auto view1 = f1.internalVector().view();
    auto view2 = f2.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        f1.exec(),
        {0, f1.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = view1[i] < view2[i] ? view1[i] : view2[i]; },
        "minFieldField"
    );
    return result;
}

void bound(VolumeField<scalar>& f, scalar lower)
{
    auto view = f.internalVector().view();

    parallelFor(
        f.exec(),
        {0, f.size()},
        NEON_LAMBDA(const localIdx i) {
            if (view[i] < lower)
            {
                view[i] = lower;
            }
        },
        "boundField"
    );
}

VolumeField<scalar> pow(const VolumeField<scalar>& f, scalar exponent)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(f.mesh());
    VolumeField<scalar> result(f.exec(), "pow", f.mesh(), bcs);

    auto inView = f.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        f.exec(),
        {0, f.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = std::pow(inView[i], exponent); },
        "powField"
    );
    return result;
}

VolumeField<Tensor> mul(const VolumeField<scalar>& s, const VolumeField<Tensor>& T)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<Tensor>>(T.mesh());
    VolumeField<Tensor> result(T.exec(), "mul", T.mesh(), bcs);

    auto sView = s.internalVector().view();
    auto tView = T.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        T.exec(),
        {0, T.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = sView[i] * tView[i]; },
        "mulScalarTensorField"
    );
    return result;
}

VolumeField<Tensor> dev2(const VolumeField<Tensor>& T)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<Tensor>>(T.mesh());
    VolumeField<Tensor> result(T.exec(), "dev2", T.mesh(), bcs);

    auto inView = T.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        T.exec(),
        {0, T.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::dev2(inView[i]); },
        "dev2TensorField"
    );
    return result;
}

VolumeField<SymmTensor> dev2(const VolumeField<SymmTensor>& S)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<SymmTensor>>(S.mesh());
    VolumeField<SymmTensor> result(S.exec(), "dev2", S.mesh(), bcs);

    auto inView = S.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        S.exec(),
        {0, S.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::dev2(inView[i]); },
        "dev2SymmTensorField"
    );
    return result;
}

VolumeField<Tensor> transpose(const VolumeField<Tensor>& T)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<Tensor>>(T.mesh());
    VolumeField<Tensor> result(T.exec(), "transpose", T.mesh(), bcs);

    auto inView = T.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        T.exec(),
        {0, T.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::T(inView[i]); },
        "transposeTensorField"
    );
    return result;
}

VolumeField<scalar> tr(const VolumeField<Tensor>& T)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(T.mesh());
    VolumeField<scalar> result(T.exec(), "tr", T.mesh(), bcs);

    auto inView = T.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        T.exec(),
        {0, T.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::tr(inView[i]); },
        "trField"
    );
    return result;
}

VolumeField<SymmTensor> devTwoSymm(const VolumeField<Tensor>& T)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<SymmTensor>>(T.mesh());
    VolumeField<SymmTensor> result(T.exec(), "devTwoSymm", T.mesh(), bcs);

    auto inView = T.internalVector().view();
    auto outView = result.internalVector().view();

    parallelFor(
        T.exec(),
        {0, T.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::devTwoSymm(inView[i]); },
        "devTwoSymmField"
    );
    return result;
}

VolumeField<scalar> doubleInner(const VolumeField<Tensor>& T, const VolumeField<SymmTensor>& S)
{
    auto bcs = createCalculatedBCs<VolumeBoundary<scalar>>(T.mesh());
    VolumeField<scalar> result(T.exec(), "doubleInner", T.mesh(), bcs);

    auto [tView, sView, outView] =
        views(T.internalVector(), S.internalVector(), result.internalVector());

    parallelFor(
        T.exec(),
        {0, T.size()},
        NEON_LAMBDA(const localIdx i) { outView[i] = NeoN::doubleInner(tView[i], sView[i]); },
        "doubleInnerField"
    );
    return result;
}

} // namespace NeoN
