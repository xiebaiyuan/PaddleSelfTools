#!/bin/bash
# The git version of CI is 2.7.4. This script is not compatible with git version 1.7.1.
set -ex
echo ">>>>"
read -r -p "Are You Sure? [Y/n] " input

case $input in
[yY][eE][sS] | [yY])
    echo "Yes"
    ;;

[nN][oO] | [nN])
    echo "No"
    exit 1
    ;;

*)
    echo "Invalid input..."
    exit 1
    ;;
esac

TESTS_FILE="./lite_tests.txt"
LIBS_FILE="./lite_libs.txt"
CUDNN_ROOT="/usr/local/cudnn"

readonly ADB_WORK_DIR="/data/local/tmp"
readonly common_flags="-DWITH_LITE=ON -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF -DWITH_PYTHON=OFF -DWITH_TESTING=ON -DLITE_WITH_ARM=OFF"

readonly THIRDPARTY_TAR=https://paddlelite-data.bj.bcebos.com/third_party_libs/third-party-ea5576.tar.gz
readonly workspace="../"

NUM_CORES_FOR_COMPILE=${LITE_BUILD_THREADS:-8}

# global variables
#whether to use emulator as adb devices,when USE_ADB_EMULATOR=ON we use emulator, else we will use connected mobile phone as adb devices.
USE_ADB_EMULATOR=OFF

# if operating in mac env, we should expand the maximum file num
os_nmae=$(uname -s)
if [ ${os_nmae} == "Darwin" ]; then
    ulimit -n 1024
fi

function prepare_thirdparty {
    cd $workspace
    if [ ! -d $workspace/third-party -o -f $workspace/third-party-ea5576.tar.gz ]; then
        rm -rf $workspace/third-party

        if [ ! -f $workspace/third-party-ea5576.tar.gz ]; then
            wget $THIRDPARTY_TAR
        fi
        tar xzf third-party-ea5576.tar.gz
    else
        git submodule update --init --recursive
    fi
    cd -
}

# for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_workspace() {
    # in build directory
    # 1. Prepare gen_code file
    GEN_CODE_PATH_PREFIX=lite/gen_code
    mkdir -p ./${GEN_CODE_PATH_PREFIX}
    touch ./${GEN_CODE_PATH_PREFIX}/__generated_code__.cc

    # 2.Prepare debug tool
    DEBUG_TOOL_PATH_PREFIX=lite/tools/debug
    mkdir -p ./${DEBUG_TOOL_PATH_PREFIX}
    cp ../${DEBUG_TOOL_PATH_PREFIX}/analysis_tool.py ./${DEBUG_TOOL_PATH_PREFIX}/

    # clone submodule
    # git submodule update --init --recursive
    prepare_thirdparty
}
function prepare_opencl_source_code {
    local root_dir=$1
    local build_dir=$2
    # in build directory
    # Prepare opencl_kernels_source.cc file
    GEN_CODE_PATH_OPENCL=$root_dir/lite/backends/opencl
    rm -f GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    OPENCL_KERNELS_PATH=$root_dir/lite/backends/opencl/cl_kernel
    mkdir -p ${GEN_CODE_PATH_OPENCL}
    touch $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    python3 $root_dir/lite/tools/cmake_tools/gen_opencl_code.py $OPENCL_KERNELS_PATH $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
}
function cmake_opencl {
    prepare_workspace
    # $1: ARM_TARGET_OS in "android" , "armlinux"
    # $2: ARM_TARGET_ARCH_ABI in "armv8", "armv7" ,"armv7hf"
    # $3: ARM_TARGET_LANG in "gcc" "clang"
    cmake .. \
        -DLITE_WITH_OPENCL=ON \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DWITH_ARM_DOTPROD=ON   \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=ON \
        -DLITE_BUILD_EXTRA=ON \
        -DLITE_WITH_LOG=ON \
        -DLITE_WITH_CV=OFF \
        -DLITE_WITH_PROFILE=ON \
        -DLITE_WITH_PRECISION_PROFILE=OFF \
        -DLITE_WITH_STATIC_CUDA=OFF \
        -DLITE_WITH_OPENMP=OFF \
        -DLITE_WITH_EXCEPTION=ON \
        -DWITH_DSO=OFF \
        -DARM_TARGET_OS=$1 -DARM_TARGET_ARCH_ABI=$2 -DARM_TARGET_LANG=$3

    # -DLITE_WITH_PROFILE=ON \

}

# -DLITE_WITH_PROFILE=ON \
# -DLITE_WITH_STATIC_CUDA=ON \

# $1: ARM_TARGET_OS in "android" , "armlinux"
# $2: ARM_TARGET_ARCH_ABI in "armv8", "armv7" ,"armv7hf"
# $3: ARM_TARGET_LANG in "gcc" "clang"
function build_opencl {
    os=$1
    abi=$2
    lang=$3

    cur_dir=$(pwd)
    if [[ ${os} == "armlinux" ]]; then
        # TODO(hongming): enable compile armv7 and armv7hf on armlinux, and clang compile
        if [[ ${lang} == "clang" ]]; then
            echo "clang is not enabled on armlinux yet"
            return 0
        fi
        if [[ ${abi} == "armv7hf" ]]; then
            echo "armv7hf is not supported on armlinux yet"
            return 0
        fi
        if [[ ${abi} == "armv7" ]]; then
            echo "armv7 is not supported on armlinux yet"
            return 0
        fi
    fi

    if [[ ${os} == "android" && ${abi} == "armv7hf" ]]; then
        echo "android do not need armv7hf"
        return 0
    fi

    build_dir=$cur_dir/build.lite.${os}.${abi}.${lang}
    mkdir -p $build_dir
    cd $build_dir

    prepare_opencl_source_code $cur_dir $build_dir

    cmake_opencl ${os} ${abi} ${lang}
#    build $TESTS_FILE
}

function build_opencl_gen_cl() {
    os=$1
    abi=$2
    lang=$3

    cur_dir=$(pwd)
    if [[ ${os} == "armlinux" ]]; then
        # TODO(hongming): enable compile armv7 and armv7hf on armlinux, and clang compile
        if [[ ${lang} == "clang" ]]; then
            echo "clang is not enabled on armlinux yet"
            return 0
        fi
        if [[ ${abi} == "armv7hf" ]]; then
            echo "armv7hf is not supported on armlinux yet"
            return 0
        fi
        if [[ ${abi} == "armv7" ]]; then
            echo "armv7 is not supported on armlinux yet"
            return 0
        fi
    fi

    if [[ ${os} == "android" && ${abi} == "armv7hf" ]]; then
        echo "android do not need armv7hf"
        return 0
    fi

    build_dir=$cur_dir/build.self.lite.${os}.${abi}.${lang}.opencl
    # rm -rf $build_dir
    # mkdir -p $build_dir
    cd $build_dir
    prepare_opencl_source_code $cur_dir $build_dir
    # cmake_opencl ${os} ${abi} ${lang}
    # make opencl_clhpp
    # build $TESTS_FILE
    # make $testname -j$NUM_CORES_FOR_COMPILE

    # # test publish inference lib
    # make publish_inference
}
function check_style() {
    export PATH=/usr/bin:$PATH
    #pre-commit install
    clang-format --version

    if ! pre-commit run -a; then
        git diff
        exit 1
    fi
}

function build_single() {
    make $1 -j$NUM_CORES_FOR_COMPILE
}

function build() {
    make lite_compile_deps -j$NUM_CORES_FOR_COMPILE
}

testname="test_nanoyolo"
rm -rf lite/api/paddle_use_kernels.h
rm -rf lite/api/paddle_use_ops.h
# build_opencl "android" "armv8" "gcc"
# build_opencl "android" "armv7" "gcc"
build_opencl "android" "armv7" "clang"
# build_opencl_gen_cl "android" "armv7" "gcc"
cd $cur
