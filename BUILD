load("//:generator.bzl", "build_example", "build_test")
load("@rules_cc//cc:defs.bzl", "cc_library")

# Shared header library used by the SVM examples.  Exposes the
# embedded Iris versicolor-vs-virginica dataset and a classification
# accuracy helper.
cc_library(
    name = "iris_data",
    hdrs = ["src/examples/iris_data.h"],
    deps = ["@eigen//:eigen"],
    visibility = ["//visibility:public"],
)

build_example("simple")
build_example("debug")
build_example("constrained_simple")
build_example("constrained_simple2")
build_example("linear_regression")
build_test("verify")
build_test("cstep_test")
build_test("hager_zhang_test")
build_test("augmented_lagrangian_test")

build_example("svm_primal_lbfgs", extra_deps = [":iris_data"])
build_example("svm_primal_al", extra_deps = [":iris_data"])
build_example("svm_dual_lbfgsb", extra_deps = [":iris_data"])
build_example("svm_dual_al", extra_deps = [":iris_data"])

