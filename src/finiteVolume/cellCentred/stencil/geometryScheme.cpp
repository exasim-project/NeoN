// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/logging.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/basicGeometryScheme.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"

#include <memory>

namespace NeoN::finiteVolume::cellCentred
{

GeometrySchemeFactory::GeometrySchemeFactory([[maybe_unused]] const UnstructuredMesh& mesh) {}


const std::shared_ptr<GeometryScheme> GeometryScheme::readOrCreate(const UnstructuredMesh& mesh)
{
    auto& db = mesh.stencilDB();
    if (!db.contains("GeometryScheme"))
    {
        db.insert(std::string("GeometryScheme"), std::make_shared<GeometryScheme>(mesh));
    }
    return db.get<std::shared_ptr<GeometryScheme>>("GeometryScheme");
}


GeometryScheme::GeometryScheme(
    const Executor& exec,
    std::unique_ptr<GeometrySchemeFactory> kernel,
    const SurfaceField<scalar>& weights,
    const SurfaceField<scalar>& deltaCoeffs,
    const SurfaceField<scalar>& nonOrthDeltaCoeffs,
    const SurfaceField<Vec3>& nonOrthCorrectionVec3s
)
    : exec_(exec), mesh_(weights.mesh()), kernel_(std::move(kernel)), weights_(weights),
      deltaCoeffs_(deltaCoeffs), nonOrthDeltaCoeffs_(nonOrthDeltaCoeffs),
      nonOrthCorrectionVec3s_(nonOrthCorrectionVec3s)
{
    if (kernel_ == nullptr)
    {
        NF_ERROR_EXIT("Kernel is not initialized");
    }
}

GeometryScheme::GeometryScheme(
    const Executor& exec,
    const UnstructuredMesh& mesh,
    std::unique_ptr<GeometrySchemeFactory> kernel
)
    : exec_(exec), mesh_(mesh), kernel_(std::move(kernel)),
      weights_(mesh.exec(), "weights", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)),
      deltaCoeffs_(
          mesh.exec(), "deltaCoeffs", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
      ),
      nonOrthDeltaCoeffs_(
          mesh.exec(),
          "nonOrthDeltaCoeffs",
          mesh,
          createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
      ),
      nonOrthCorrectionVec3s_(
          mesh.exec(),
          "nonOrthCorrectionVec3s",
          mesh,
          createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
      )
{
    if (kernel_ == nullptr)
    {
        NF_ERROR_EXIT("Kernel is not initialized");
    }
    update();
}

GeometryScheme::GeometryScheme(const UnstructuredMesh& mesh)
    : exec_(mesh.exec()), mesh_(mesh),
      kernel_(std::make_unique<BasicGeometryScheme>(mesh)), // TODO add selection mechanism
      weights_(mesh.exec(), "weights", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)),
      deltaCoeffs_(
          mesh.exec(), "deltaCoeffs", mesh, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
      ),
      nonOrthDeltaCoeffs_(
          mesh.exec(),
          "nonOrthDeltaCoeffs",
          mesh,
          createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
      ),
      nonOrthCorrectionVec3s_(
          mesh.exec(),
          "nonOrthCorrectionVec3s",
          mesh,
          createCalculatedBCs<SurfaceBoundary<Vec3>>(mesh)
      )
{
    if (kernel_ == nullptr)
    {
        NF_ERROR_EXIT("Kernel is not initialized");
    }
    update();
}

std::string GeometryScheme::name() const { return std::string("GeometryScheme"); }

void GeometryScheme::update()
{
    if (mesh_.faceCenters().size() > 0)
    {
        std::visit(
            [&](const auto& exec)
            {
                kernel_->updateWeights(exec, weights_);
                kernel_->updateDeltaCoeffs(exec, deltaCoeffs_);
                kernel_->updateNonOrthDeltaCoeffs(exec, nonOrthDeltaCoeffs_);
                kernel_->updateNonOrthCorrectionVec3s(exec, nonOrthCorrectionVec3s_);
            },
            exec_
        );
        reset();
    }
}

void GeometryScheme::reset() const
{
    // TODO this needs a better approach
    // ideally faceCenters are some kind of hostViewVector
    Logging::warn("resetting face and cell centers");
    const_cast<UnstructuredMesh&>(mesh_).faceCenters().resize(0);
    const_cast<UnstructuredMesh&>(mesh_).cellCenters().resize(0);
}


const SurfaceField<scalar>& GeometryScheme::weights() const { return weights_; }

const SurfaceField<scalar>& GeometryScheme::deltaCoeffs() const { return deltaCoeffs_; }

const SurfaceField<scalar>& GeometryScheme::nonOrthDeltaCoeffs() const
{
    return nonOrthDeltaCoeffs_;
}

const SurfaceField<Vec3>& GeometryScheme::nonOrthCorrectionVec3s() const
{
    return nonOrthCorrectionVec3s_;
}

} // namespace NeoN
