#!/bin/bash
set -e # Use Bash strict mode

printf "Build fmt\n"
./build_using_cmake.sh fmt -DFMT_DOC=OFF -DFMT_TEST=OFF -DFMT_HEADER_ONLY=ON >/dev/null

printf "Build benchmark\n"
./build_using_cmake.sh benchmark -DBENCHMARK_ENABLE_TESTING=OFF >/dev/null

printf "Build doctest\n"
cp -rv src/doctest/doctest include/

printf "Build eigen\n"
./build_using_cmake.sh eigen
