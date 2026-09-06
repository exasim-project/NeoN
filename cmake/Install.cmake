# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

install(
  TARGETS NeoN NeoN_public_api NeoN_options NeoN_warnings
  EXPORT ${PROJECT_NAME}Targets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Add Imported packages via CPM (FetchContent) to export set of neon

if(ginkgo_POPULATED)
  install(
    TARGETS ginkgo
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES
    DESTINATION include)
endif()

if(nlohmann_json_POPULATED)
  install(
    TARGETS nlohmann_json
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES
    DESTINATION include)
endif()

# umpire installs its own targets when fetched via CPM

# install internal cpptrace version
if(cpptrace_POPULATED)
  install(
    TARGETS cpptrace
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES
    DESTINATION include)
endif()

# install internal spdlog version TODO why does spdlog_POPULATED not work?
if(spdlog_ADDED)
  install(
    TARGETS spdlog_header_only
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES
    DESTINATION include)
endif()

if(fmt_ADDED)
  install(
    TARGETS fmt fmt-header-only
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES
    DESTINATION include)
endif()

# install internal sundials version if(sundials_POPULATED)
if(sundials_ADDED)
  install(
    TARGETS cvode nvec arkode sundials_core_shared
    EXPORT SundialsTargets
    INCLUDES
    DESTINATION include)
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeoNConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/NeoNConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeoN)

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE NeoNTargets.cmake
  NAMESPACE NeoN::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeoN)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/NeoNConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeoN)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/NeoN/NeoN.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/NeoN)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/NeoN/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/NeoN)
