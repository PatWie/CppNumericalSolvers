package(default_visibility = ["//visibility:public"])

load("//:generator.bzl", "build_example", "build_test")

build_example("linearregression")
build_example("logisticregression")
build_example("rosenbrock")
build_example("rosenbrock_float")
build_example("simple")
build_example("simple_withoptions")
build_example("neldermead")
build_example("neldermead-customized")
build_example("nonnegls")
build_example("nonnegls2")

build_test("verify")


py_binary(
    name = "lint",
    args = ["src/test/verify.cpp"],
    srcs = ["lint.py"],
)