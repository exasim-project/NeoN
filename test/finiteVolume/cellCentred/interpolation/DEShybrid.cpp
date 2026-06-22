// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using NeoN::finiteVolume::cellCentred::SurfaceInterpolation;
using NeoN::finiteVolume::cellCentred::VolumeField;
using NeoN::finiteVolume::cellCentred::SurfaceField;

namespace NeoN
{

namespace fvcc = finiteVolume::cellCentred;

template<typename T>
using I = std::initializer_list<T>;

// Builds a source volume field f(x) = a.x + d (scalar) / per-component linear law (Vec3) on the
// uniform mesh so both linear and linearUpwind produce well-defined, distinct face values.
template<typename TestType>
VolumeField<TestType> makeLinearField(const NeoN::Executor& exec, const UnstructuredMesh& mesh)
{
    auto ccH = mesh.cellCenters().copyToHost();
    const auto cc = ccH.view();
    auto src = VolumeField<TestType>(
        exec, "src", mesh, fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh)
    );
    auto hostIn = src.internalVector().copyToHost();
    for (localIdx i = 0; i < hostIn.size(); ++i)
    {
        const Vec3 p = cc[i];
        if constexpr (std::is_same_v<TestType, scalar>)
        {
            hostIn.view()[i] = 1.5 * p[0] - 2.0 * p[1] + 0.75 * p[2] + 0.3;
        }
        else
        {
            hostIn.view()[i] = Vec3 {
                1.5 * p[0] - 2.0 * p[1] + 0.75 * p[2] + 0.3,
                -0.5 * p[0] + 1.25 * p[1] - 0.4 * p[2] - 0.1,
                2.0 * p[0] + 0.6 * p[1] - 1.1 * p[2] + 0.2
            };
        }
    }
    src.internalVector() = hostIn.copyToExecutor(exec);
    src.correctBoundaryConditions();
    return src;
}

template<typename TestType>
SurfaceField<scalar>
makeUniformSigma(const NeoN::Executor& exec, const UnstructuredMesh& mesh, scalar value)
{
    auto sigma = SurfaceField<scalar>(
        exec, "sigma", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );
    fill(sigma.internalVector(), value);
    fill(sigma.boundaryData().value(), value);
    return sigma;
}

// The DEShybrid scheme must be registered under "DEShybrid", parse its two sub-schemes, and with
// the default (zero) blending factor reduce exactly to scheme 1 (linear).
TEMPLATE_TEST_CASE("DEShybrid registered; default sigma=0 reduces to scheme 1", "", scalar, Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    auto mesh = create3DUniformMesh(exec, 4, 4, 4);
    auto src = makeLinearField<TestType>(exec, mesh);

    Input desInput =
        TokenList({std::string("DEShybrid"), std::string("linear"), std::string("linearUpwind")});
    auto des = SurfaceInterpolation<TestType>(exec, mesh, desInput);

    Input linInput = TokenList({std::string("linear")});
    auto lin = SurfaceInterpolation<TestType>(exec, mesh, linInput);

    auto flux = SurfaceField<scalar>(
        exec, "flux", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );
    fill(flux.internalVector(), one<scalar>());

    auto outDes = SurfaceField<TestType>(
        exec, "outDes", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh)
    );
    auto outLin = SurfaceField<TestType>(
        exec, "outLin", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh)
    );

    des.interpolate(flux, src, outDes);
    lin.interpolate(flux, src, outLin);

    auto dH = outDes.internalVector().copyToHost();
    auto lH = outLin.internalVector().copyToHost();
    for (localIdx i = 0; i < mesh.nInternalFaces(); ++i)
    {
        REQUIRE(mag(dH.view()[i] - lH.view()[i]) < 1e-12);
    }
}

// With a uniform blending factor sigma, the DEShybrid face value, weights and correction must equal
// the explicit elementwise blend (1-sigma)*scheme1 + sigma*scheme2 of the two sub-schemes —
// including the limits sigma=1 (== scheme 2, linearUpwind).
TEMPLATE_TEST_CASE("DEShybrid blends scheme1 and scheme2 by sigma", "", scalar, Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());
    INFO("executor: " << execName);

    const scalar sigmaVal = GENERATE(scalar(0.0), scalar(0.3), scalar(1.0));
    INFO("sigma: " << sigmaVal);

    auto mesh = create3DUniformMesh(exec, 4, 4, 4);
    auto src = makeLinearField<TestType>(exec, mesh);

    // sub-schemes evaluated independently
    auto lin =
        SurfaceInterpolation<TestType>(exec, mesh, Input(TokenList({std::string("linear")})));
    auto lup = SurfaceInterpolation<TestType>(
        exec, mesh, Input(TokenList({std::string("linearUpwind"), std::string("Gauss")}))
    );

    // DEShybrid constructed directly so the (test) blending factor can be set explicitly
    fvcc::DEShybrid<TestType> des(
        exec, mesh, Input(TokenList({std::string("linear"), std::string("linearUpwind")}))
    );
    auto sigma = makeUniformSigma<TestType>(exec, mesh, sigmaVal);
    des.setBlendingFactor(sigma);

    REQUIRE(des.corrected()); // scheme 2 (linearUpwind) carries a correction

    auto flux = SurfaceField<scalar>(
        exec, "flux", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );
    fill(flux.internalVector(), one<scalar>());

    auto make = [&](const std::string& nm)
    {
        return SurfaceField<TestType>(
            exec, nm, mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<TestType>>(mesh)
        );
    };

    // ---- interpolate ----
    auto i1 = make("i1");
    auto i2 = make("i2");
    auto iDes = make("iDes");
    lin.interpolate(flux, src, i1);
    lup.interpolate(flux, src, i2);
    des.interpolate(flux, src, iDes);

    auto i1H = i1.internalVector().copyToHost();
    auto i2H = i2.internalVector().copyToHost();
    auto iDesH = iDes.internalVector().copyToHost();
    for (localIdx i = 0; i < mesh.nInternalFaces(); ++i)
    {
        const TestType expected = (scalar(1) - sigmaVal) * i1H.view()[i] + sigmaVal * i2H.view()[i];
        REQUIRE(mag(iDesH.view()[i] - expected) < 1e-12);
    }

    // ---- weights ----
    auto w1 = SurfaceField<scalar>(
        exec, "w1", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );
    auto w2 = SurfaceField<scalar>(
        exec, "w2", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );
    auto wDes = SurfaceField<scalar>(
        exec, "wDes", mesh, fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh)
    );
    lin.weight(flux, src, w1);
    lup.weight(flux, src, w2);
    des.weight(flux, src, wDes);

    auto w1H = w1.internalVector().copyToHost();
    auto w2H = w2.internalVector().copyToHost();
    auto wDesH = wDes.internalVector().copyToHost();
    for (localIdx i = 0; i < mesh.nInternalFaces(); ++i)
    {
        const scalar expected = (scalar(1) - sigmaVal) * w1H.view()[i] + sigmaVal * w2H.view()[i];
        REQUIRE(wDesH.view()[i] == Catch::Approx(expected).margin(1e-12));
    }

    // ---- correction (scheme1 linear is uncorrected -> contributes zero) ----
    auto c2 = make("c2");
    auto cDes = make("cDes");
    lup.correction(flux, src, c2);
    des.correction(flux, src, cDes);

    auto c2H = c2.internalVector().copyToHost();
    auto cDesH = cDes.internalVector().copyToHost();
    for (localIdx i = 0; i < mesh.nInternalFaces(); ++i)
    {
        const TestType expected = sigmaVal * c2H.view()[i]; // (1-sigma)*0 + sigma*corr2
        REQUIRE(mag(cDesH.view()[i] - expected) < 1e-12);
    }
}

} // namespace NeoN
