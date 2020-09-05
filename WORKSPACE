load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

# Load 3.3.9
new_git_repository(
  name="eigen_archive",
  commit="5f25bcf7d6918f5c6091fb4e961e5607e13b7324",
  remote="https://gitlab.com/libeigen/eigen.git",
  build_file = "@//:eigen.BUILD",
)

http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "@//:gtest.BUILD",
    strip_prefix = "googletest-release-1.7.0",
)
