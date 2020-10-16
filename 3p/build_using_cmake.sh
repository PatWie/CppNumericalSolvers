#!/bin/bash
set -euo pipefail # Use Bash strict mode

PKGNAME=$1
CMAKE_OPTIONS=${*:2}

osType=$(uname)
case "$osType" in
    "Darwin")
        {
            NCPUS=$(sysctl -n hw.ncpu)
            BUILD_OPTS=-j$((NCPUS + 1))
        }
        ;;
    "Linux")
        {
            NCPUS=$(grep -c ^processor /proc/cpuinfo)
            BUILD_OPTS=-j$((NCPUS + 1))
        }
        ;;
    *)
        {
            echo "Unsupported OS, exiting"
            exit
        }
        ;;
esac

ROOT_DIR="$PWD"
SRC_DIR=$ROOT_DIR/src
TMP_DIR=/tmp/build/
PREFIX=$ROOT_DIR
mkdir -p $TMP_DIR

APKG_SRC=$SRC_DIR/$PKGNAME
APKG_BUILD_FOLDER=$TMP_DIR/$PKGNAME
APKG_PREFIX=$PREFIX

echo "Src folder: " "$APKG_SRC"
echo "Build folder: " "$APKG_BUILD_FOLDER"
echo "Prefix folder: " "$APKG_PREFIX"

# Build a given package
rm -rf "$APKG_BUILD_FOLDER"
mkdir -p "$APKG_BUILD_FOLDER"

pushd "$APKG_BUILD_FOLDER"
set -x

# shellcheck disable=SC2086
cmake "$APKG_SRC" -DCMAKE_INSTALL_PREFIX="$APKG_PREFIX" $CMAKE_OPTIONS

make "$BUILD_OPTS"
make install
set +x
# Return to the original folder.
popd

# Cleanup build folder
rm -rf "$APKG_BUILD_FOLDER"
