# Function to build tests
function(build_test name)
  add_executable(${name} ${name}.cc)
  target_compile_options(${name} PRIVATE -std=c++17 -Wall -Wextra)
  target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE CppNumericalSolvers Eigen3::Eigen GTest::GTest GTest::Main)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Find Eigen and GoogleTest
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# Build tests
build_test(verify)


