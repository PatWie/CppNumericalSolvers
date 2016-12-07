cmake_minimum_required(VERSION 3.1)
project(CppNumericalSolvers)

option(BUILD_TESTING "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_TEST_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

# check existence of c++11 compiler
if(NOT CMAKE_CXX_STANDARD) # don't override
    set(CMAKE_CXX_STANDARD 11)
endif()

# set path to cmake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(LINKER_LIBS "")

if(BUILD_TESTING)
    # Google Testing Framework
    add_subdirectory(gtest)
    include_directories(gtest)
endif()

# Eigen3 Headerfiles, prefer config-module
find_package(Eigen3 CONFIG QUIET) 

# fall-back to find-module
if(NOT Eigen3_FOUND AND NOT EIGEN3_FOUND)
    find_package(Eigen3 REQUIRED)
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
else()
    link_libraries(::Eigen3)
endif()

add_library(cppoptlib INTERFACE)
target_include_directories(cppoptlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
if(TARGET ::Eigen3)
    target_link_libraries(cppoptlib INTERFACE ::Eigen3)
endif()

install(TARGETS cppoptlib EXPORT cppoptlib-targets)

if(TARGET ::Eigen3)
    install(EXPORT cppoptlib-targets DESTINATION lib/cmake/cppoptlib
        NAMESPACE ::) # export name will be '::cppoptlib'
    # write and install config module
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hide/cppoptlib-config.cmake "
        include(CMakeFindDependencyMacro)
        find_dependency(Eigen3)
        if(Eigen3_FOUND)
            include(\"\${CMAKE_CURRENT_LIST_DIR}/cppoptlib-targets.cmake\")
        endif()
    ")
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hide/cppoptlib-config.cmake
        DESTINATION lib/cmake/cppoptlib)
else()
    # config-module without referencing Eigen
    install(EXPORT cppoptlib-targets DESTINATION lib/cmake/cppoptlib
        NAMESPACE :: # export name will be '::cppoptlib'
        FILE cppoptlib-config.cmake)
endif()

install(DIRECTORY include/cppoptlib DESTINATION include)

if(BUILD_EXAMPLES)
    add_subdirectory(src/examples)
endif()

if(BUILD_TESTING)
    add_subdirectory(src/test)
endif()

add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)
