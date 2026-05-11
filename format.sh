#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/sprout/dist/gcc/lib64

CLANG_FORMAT=/sprout/dist/clang/bin/clang-format

find include src -type f \( -name "*.h" -o -name "*.cc" -o -name "*.cpp" \) \
  -exec "$CLANG_FORMAT" -i --style=Google {} +
