# Function to build examples
function(build_example name)
  add_executable(${name} ${name}.cc)
  target_compile_options(${name} PRIVATE -std=c++17 -Wall -Wextra)
  target_link_libraries(${name} PRIVATE CppNumericalSolvers Eigen3::Eigen)
endfunction()

# Find Eigen and GoogleTest
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

# Build examples
build_example(simple)
build_example(constrained_simple)
