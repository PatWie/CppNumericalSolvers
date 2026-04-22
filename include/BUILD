load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "cppoptlib",
    hdrs = glob([
        "cppoptlib/*.h",
        "cppoptlib/solver/*.h",
        "cppoptlib/linesearch/*.h",
        "cppoptlib/utils/*.h",
    ]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
