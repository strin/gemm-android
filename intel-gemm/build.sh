#!/usr/bin/env sh
set -e

if [ -z "$NDK_ROOT" ] && [ "$#" -eq 0 ]; then
    echo 'Either $NDK_ROOT should be set or provided as argument'
    echo "e.g., 'export NDK_ROOT=/path/to/ndk' or"
    echo "      '${0} /path/to/ndk'"
    exit 1
else
    NDK_ROOT="${1:-${NDK_ROOT}}"
fi

ANDROID_ABI=${ANDROID_ABI:-"armeabi-v7a-hard with NEON"}
WD=`pwd`
N_JOBS=${N_JOBS:-4}
GEMM_ROOT=${WD}/intel-gemm
BUILD_DIR=${GEMM_ROOT}/build
ANDROID_LIB_ROOT=${WD}/
OPENCV_ROOT=${ANDROID_LIB_ROOT}/opencv/sdk/native/jni
OPENCL_ROOT=${ANDROID_LIB_ROOT}/opencl

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -DCMAKE_TOOLCHAIN_FILE="${WD}/android-cmake/android.toolchain.cmake" \
      -DANDROID_NDK="${NDK_ROOT}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="${ANDROID_ABI}" \
      -DANDROID_NATIVE_API_LEVEL=21 \
      -DOPENCL_LIBS=${OPENCL_ROOT}/libOpenCL.so \
      -DOPENCL_INCLUDES=${OPENCL_ROOT} \
      ..

make -j${N_JOBS}

cd "${WD}"
