cmake_minimum_required(VERSION 3.12)
project(CppNumericalSolvers VERSION 1.0.0 LANGUAGES CXX)

add_library(CppNumericalSolvers INTERFACE)

option(BUILD_EXAMPLES "Build the examples" ON)
option(BUILD_TESTS "Build the tests" ON)

target_include_directories(CppNumericalSolvers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(CppNumericalSolvers INTERFACE cxx_std_11)

# Install the target
install(TARGETS CppNumericalSolvers
  EXPORT CppNumericalSolversTargets
  INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/ DESTINATION include)

# Install the CMake package configuration
install(EXPORT CppNumericalSolversTargets
  FILE CppNumericalSolversTargets.cmake
  NAMESPACE CppNumericalSolvers::
  DESTINATION lib/cmake/CppNumericalSolvers
)

# Generate a Config file for find_package()
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CppNumericalSolversConfig.cmake
  INSTALL_DESTINATION lib/cmake/CppNumericalSolvers
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CppNumericalSolversConfig.cmake
  DESTINATION lib/cmake/CppNumericalSolvers
)

if(BUILD_EXAMPLES)
  add_subdirectory(src/examples)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(src/test)
endif()
