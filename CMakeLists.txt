cmake_minimum_required(VERSION 3.12)
project(CppNumericalSolvers VERSION 1.0.0 LANGUAGES CXX)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

add_library(CppNumericalSolvers INTERFACE)

option(CppNumericalSolvers_INSTALL "Generate the install target." ON)
option(BUILD_EXAMPLES "Build the examples" OFF)
option(BUILD_TESTS "Build the tests" OFF)

target_include_directories(CppNumericalSolvers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(CppNumericalSolvers INTERFACE cxx_std_11)

if(CppNumericalSolvers_INSTALL)
  # Install the interface target and export it.
  install(TARGETS CppNumericalSolvers
    EXPORT CppNumericalSolversTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  # Install header files (everything under the include/ folder)
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # Export the target so that consumers can use find_package.
  install(EXPORT CppNumericalSolversTargets
    FILE CppNumericalSolversTargets.cmake
    NAMESPACE CppNumericalSolvers::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CppNumericalSolvers
  )

  # Generate the package configuration file from a template.
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CppNumericalSolversConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CppNumericalSolvers
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )

  # Generate the version file.
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CppNumericalSolversConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  # Install the generated config files.
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/CppNumericalSolversConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/CppNumericalSolversConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CppNumericalSolvers
  )

  # Export the target for use from the build tree.
  export(TARGETS CppNumericalSolvers NAMESPACE CppNumericalSolvers:: FILE CppNumericalSolversTargets.cmake)
  export(PACKAGE CppNumericalSolvers)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(src/examples)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(src/test)
endif()
