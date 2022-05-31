#!/usr/bin/env bash
set -o errexit
function prepare_opencl_source_code() {
    local root_dir=$1
    echo "update cl kernel..."
    GEN_CODE_PATH_OPENCL=$root_dir/lite/backends/opencl
    rm -f GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    OPENCL_KERNELS_PATH=$root_dir/lite/backends/opencl/cl_kernel
    mkdir -p ${GEN_CODE_PATH_OPENCL}
    touch $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    python3 $root_dir/lite/tools/cmake_tools/gen_opencl_code.py $OPENCL_KERNELS_PATH $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
}

with_cmake=false
with_make=true
with_push=false

testname="test_expand_image_opencl"

echo "with cmake : $with_cmake"
echo "with make : $with_make"
echo "testname : $testname"

if [[ "$with_cmake" == "true" ]]; then
    cd PaddleMobileTools
    ./ci_build_cmake.sh
    cd -
fi

if [[ "$with_make" == "true" ]]; then
    prepare_opencl_source_code $(pwd)
    cd build.self.lite.android.armv7.gcc.opencl
    make $testname -j$6
    cd -
fi

adb push build.self.lite.android.armv7.gcc.opencl/lite/kernels/opencl/${testname} /data/local/tmp/opencl/${testname}
adb shell "export GLOG_v=10; /data/local/tmp/opencl/${testname}"
