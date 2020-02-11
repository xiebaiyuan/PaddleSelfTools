#!/usr/bin/env bash
set -e
function print_usage() {
  echo "\n${RED}Usage${NONE}:
  ${BOLD}${SCRIPT_NAME}${NONE} [Option] [Network]"

  echo "\n${RED}Option${NONE}: required, specify the target platform
  ${BLUE}android_armv7${NONE}: run build for android armv7 platform
  ${BLUE}android_armv8${NONE}: run build for android armv8 platform
  ${BLUE}ios${NONE}: run build for apple ios platform
  ${BLUE}linux_armv7${NONE}: run build for linux armv7 platform
  ${BLUE}linux_armv8${NONE}: run build for linux armv8 platform
  ${BLUE}fpga${NONE}: run build for fpga platform
  "
  echo "\n${RED}Network${NONE}: optional, for deep compressing the framework size
  ${BLUE}googlenet${NONE}: build only googlenet support
  ${BLUE}mobilenet${NONE}: build only mobilenet support
  ${BLUE}yolo${NONE}: build only yolo support
  ${BLUE}squeezenet${NONE}: build only squeezenet support
  ${BLUE}resnet${NONE}: build only resnet support
  ${BLUE}mobilenetssd${NONE}: build only mobilenetssd support
  ${BLUE}nlp${NONE}: build only nlp model support
  ${BLUE}mobilenetfssd${NONE}: build only mobilenetfssd support
  ${BLUE}genet${NONE}: build only genet support
  ${BLUE}super${NONE}: build only super support
  "
}

function init() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  PADDLE_MOBILE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
  if [ -z "${SCRIPT_NAME}" ]; then
    SCRIPT_NAME=$0
  fi

  rm -rf ../build/package_old
  mv ../build/package ../build/package_old
  # rm -rf ../build/package
  mkdir ../build/package
  cd ../build/package/
  rm -rf gitinfo.txt
  date +%Y-%m-%d-%H:%M:%S >>gitinfo.txt
  echo "current commit id: " >>gitinfo.txt
  git rev-parse HEAD >>gitinfo.txt
  cd -

  # merge cl to so
  merge_cl_to_so=1
  opencl_kernels="opencl_kernels.cpp"
  cd ../src/operators/kernel/cl
  if [[ -f "${opencl_kernels}" ]]; then
    rm "${opencl_kernels}"
  fi
  python gen_code.py "${merge_cl_to_so}" >"${opencl_kernels}"
  cd -

  # get cl headers
  # opencl_header_dir="../third_party/opencl/OpenCL-Headers"
  # commit_id="320d7189b3e0e7b6a8fc5c10334c79ef364b5ef6"
  # if [[ -d "$opencl_header_dir" && -d "$opencl_header_dir/.git" ]]; then
  #     echo "pulling opencl headers"
  #     cd $opencl_header_dir
  #     git stash
  #     git pull
  #     git checkout $commit_id
  #     cd -
  # else
  #     echo "cloning opencl headers"
  #     rm -rf $opencl_header_dir
  #     git clone https://github.com/KhronosGroup/OpenCL-Headers $opencl_header_dir
  #     git checkout $commit_id
  # fi
}

function check_ndk() {
  if [ -z "${NDK_ROOT}" ]; then
    echo "Should set NDK_ROOT as your android ndk path, such as\n"
    echo "  export NDK_ROOT=~/android-ndk-r14b\n"
    exit -1
  fi
}

function build_android_armv7_cpu_only() {
  # rm -rf ../build/armeabi-v7a-cpu
  cmake .. \
    -B"../build/armeabi-v7a-cpu" \
    -DANDROID_ABI="armeabi-v7a with NEON" \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake" \
    -DANDROID_PLATFORM="android-19" \
    -DANDROID_STL=c++_static \
    -DANDROID=true \
    -DWITH_LOGGING=OFF \
    -DWITH_PROFILE=OFF \
    -DUSE_EXCEPTION=ON \
    -DUSE_OPENMP=OFF \
    -DWITH_SYMBOL=ON \
    -DCPU=ON \
    -DGPU_CL=OFF \
    -DPREPARE_OPENCL_RUNTIME=ON \
    -DFPGA=OFF

  cd ../build/armeabi-v7a-cpu && make -j 8
  cd -

}

function build_android_armv7_gpu() {
  # rm -rf ../build/armeabi-v7a-gpu
  cmake .. \
    -B"../build/armeabi-v7a-gpu" \
    -DANDROID_ABI="armeabi-v7a with NEON" \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake" \
    -DANDROID_PLATFORM="android-19" \
    -DANDROID_STL=c++_static \
    -DANDROID=true \
    -DWITH_LOGGING=OFF \
    -DWITH_PROFILE=OFF \
    -DUSE_EXCEPTION=ON \
    -DUSE_OPENMP=OFF \
    -DWITH_SYMBOL=ON \
    -DCPU=ON \
    -DGPU_CL=ON \
    -DPREPARE_OPENCL_RUNTIME=ON \
    -DFPGA=OFF

  cd ../build/armeabi-v7a-gpu && make -j 8
  cd -
}

function build_android_armv8_cpu_only() {
  # rm -rf ../build/arm64-v8a-cpu
  cmake .. \
    -B"../build/arm64-v8a-cpu" \
    -DANDROID_ABI="arm64-v8a" \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake" \
    -DANDROID_PLATFORM="android-19" \
    -DANDROID_STL=c++_static \
    -DANDROID=true \
    -DWITH_LOGGING=OFF \
    -DWITH_PROFILE=OFF \
    -DUSE_EXCEPTION=ON \
    -DUSE_OPENMP=OFF \
    -DWITH_SYMBOL=ON \
    -DCPU=ON \
    -DGPU_CL=OFF \
    -DPREPARE_OPENCL_RUNTIME=ON \
    -DFPGA=OFF
  cd ../build/arm64-v8a-cpu && make -j 1
  cd -
}
# MinSizeRel
function build_android_armv8_gpu() {
  # rm -rf ../build/arm64-v8a-gpu
  cmake .. \
    -B"../build/arm64-v8a-gpu" \
    -DANDROID_ABI="arm64-v8a" \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake" \
    -DANDROID_PLATFORM="android-19" \
    -DANDROID_STL=c++_static \
    -DANDROID=true \
    -DWITH_LOGGING=OFF \
    -DWITH_PROFILE=OFF \
    -DUSE_EXCEPTION=ON \
    -DUSE_OPENMP=OFF \
    -DWITH_SYMBOL=ON \
    -DCPU=ON \
    -DGPU_CL=ON \
    -DPREPARE_OPENCL_RUNTIME=ON \
    -DFPGA=OFF

  cd ../build/arm64-v8a-gpu && make -j 8
  cd -
}

function build_ios_armv8_cpu_only() {
  # rm -rf ../build/ios
  cmake .. \
    -B"../build/ios" \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/ios-cmake/ios.toolchain.cmake" \
    -DIOS_PLATFORM=OS \
    -DIOS_ARCH="${IOS_ARCH}" \
    -DIS_IOS=true \
    -DUSE_OPENMP=OFF \
    -DCPU=ON \
    -DGPU_CL=OFF \
    -DFPGA=OFF

  cd ../build/ios && make -j 8
  cd -
}

function build_ios_armv8_gpu() {
  # rm -rf ../build/ios
  cmake .. \
    -B"../build/ios" \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/ios-cmake/ios.toolchain.cmake" \
    -DIOS_PLATFORM=OS \
    -DIOS_ARCH="${IOS_ARCH}" \
    -DIS_IOS=true \
    -DUSE_OPENMP=OFF \
    -DCPU=ON \
    -DGPU_CL=ON \
    -DFPGA=OFF

  cd ../build/ios && make -j 8
  cd -
}

function build_linux_armv7_cpu_only() {
  rm -rf ../build/armv7_linux
  cmake .. \
    -B"../build/armv7_linux" \
    -DCMAKE_BUILD_TYPE="MinSizeRel" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/toolchains/arm-linux-gnueabihf.cmake" \
    -DCPU=ON \
    -DGPU_CL=OFF \
    -DFPGA=OFF

  cd ../build/armv7_linux && make -j 8
  cd -
}

function build_linux_armv7_gpu() {
  rm -rf ../build/armv7_linux
  cmake .. \
    -B"../build/armv7_linux" \
    -DCMAKE_BUILD_TYPE="MinSizeRel" \
    -DCMAKE_TOOLCHAIN_FILE="./tools/toolchains/arm-linux-gnueabihf.cmake" \
    -DCPU=ON \
    -DGPU_CL=ON \
    -DFPGA=OFF

  cd ../build/armv7_linux && make -j 8
  cd -
}

function build_android_armv7() {
  check_ndk
  build_android_armv7_cpu_only
  build_android_armv7_gpu
}

function build_android_armv8() {
  check_ndk
  build_android_armv8_cpu_only
  build_android_armv8_gpu
}

function build_ios() {
  build_ios_armv8_cpu_only
  # build_ios_armv8_gpu
}

function build_linux_armv7() {
  build_linux_armv7_cpu_only
  # build_linux_armv7_gpu
}

function build_linux_fpga() {
  cd ..
  image=$(docker images paddle-mobile:dev | grep 'paddle-mobile')
  if [[ "x"$image == "x" ]]; then
    docker build -t paddle-mobile:dev - <Dockerfile
  fi
  docker run --rm -v $(pwd):/workspace paddle-mobile:dev bash /workspace/tools/docker_build_fpga.sh
  cd -
}

function run_android_test() {
  ExecuteAndroidTests $1
}

function cp_asserts() {
  cp ../build/arm64-v8a-cpu/build/libpaddle-mobile.so ../build/package/libpaddle-mobile-v8a-cpu.so
  cp ../build/arm64-v8a-gpu/build/libpaddle-mobile.so ../build/package/libpaddle-mobile-v8a-gpu.so
  cp ../build/armeabi-v7a-cpu/build/libpaddle-mobile.so ../build/package/libpaddle-mobile-v7a-cpu.so
  cp ../build/armeabi-v7a-gpu/build/libpaddle-mobile.so ../build/package/libpaddle-mobile-v7a-gpu.so

  cp ../src/io/paddle_inference_api.h ../build/package/paddle_inference_api.h

  cd "../build/"
  now=$(date +%Y-%m-%d-%H:%M:%S)
  echo $now
  zip -r paddlemobile_$now ./package/
  cd -

}

function main() {
  local CMD=$1
  init
  case $CMD in
  android_armv7)
    build_android_armv7
    # run_android_test armeabi-v7a
    ;;
  android_armv8)
    build_android_armv8
    # run_android_test arm64-v8a
    ;;
  ios)
    build_ios
    ;;
  linux_armv7)
    build_linux_armv7
    ;;
  fpga)
    build_linux_fpga
    ;;
  *)
    print_usage
    exit 0
    ;;
  esac
}
# main $@
# main android_armv7
# main android_armv8
#  export NDK_ROOT=/opt/android-ndk-r17c

# init
main android_armv7
main android_armv8
cp_asserts
