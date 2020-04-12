#!/usr/bin/env bash
set -o errexit
function prepare_opencl_source_code() {
    local root_dir=$1
    # local build_dir=$2
    # in build directory
    # Prepare opencl_kernels_source.cc file
    echo "update cl kernel..."
    GEN_CODE_PATH_OPENCL=$root_dir/lite/backends/opencl
    rm -f GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    OPENCL_KERNELS_PATH=$root_dir/lite/backends/opencl/cl_kernel
    mkdir -p ${GEN_CODE_PATH_OPENCL}
    touch $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
    python $root_dir/lite/tools/cmake_tools/gen_opencl_code.py $OPENCL_KERNELS_PATH $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
}

with_cmake=false
with_make=true
with_push=true

# input_dir="/data/coremodels/Lens_YoloNano/feeds/"
# output_dir="/data/coremodels/Lens_YoloNano/outputs/"
# input="image"
# output="save_infer_model_scale_0"
# source_model_dir="/data/coremodels/Lens_YoloNano/checked_model/saved-20200222-215656"
# model_dir="/data/local/tmp/opencl/models/nanoyolo/"
testname="test_layout_image_opencl"

echo "with cmake : $with_cmake"
echo "with_make : $with_make"
echo "with_push : $with_push"
# echo "input_dir : $input_dir"
# echo "output_dir : $output_dir"
# echo "input : $input"
# echo "output : $output"
# echo "source_model_dir : $source_model_dir"
# echo "model_dir : $model_dir"
echo "testname : $testname"

pwd
if [[ "$with_cmake" == "true" ]]; then
    cd PaddleMobileTools
    ./ci_build_cmake.sh
    cd -
fi

if [[ "$with_make" == "true" ]]; then
    prepare_opencl_source_code $(pwd)
    cd build.self.lite.android.armv7.clang.opencl
    make $testname -j$6
    cd -
fi

if [[ "$with_push" == "true" ]]; then
    # 在/data/local/tmp目录下创建OpenCL文件目录
    # adb shell mkdir -p /data/local/tmp/opencl
    # adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/buffer
    # adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/image

    # # 将OpenCL的kernels文件推送到/data/local/tmp/opencl目录下
    # adb push lite/backends/opencl/cl_kernel/cl_common.h /data/local/tmp/opencl/cl_kernel/
    # adb push lite/backends/opencl/cl_kernel/buffer/* /data/local/tmp/opencl/cl_kernel/buffer/
    # adb push lite/backends/opencl/cl_kernel/image/* /data/local/tmp/opencl/cl_kernel/image/
    echo "with push"
    # adb shell mkdir -p ${model_dir}
    # adb push ${input_dir}${input} /data/local/tmp/opencl/${input}
    # adb push ${output_dir}${output} /data/local/tmp/opencl/${output}
    # adb push ${source_model_dir}/* ${model_dir}
fi

# adb shell mkdir -p /data/local/tmp/opencl/mobilenet_v1
#adb push build.lite.android.armv8.gcc.opencl/third_party/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1/

# 将OpenCL单元测试程序test_mobilenetv1，推送到/data/local/tmp/opencl目录下
#adb push build.lite.android.armv8.gcc.opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl

#adb push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_layout_opencl /data/local/tmp/opencl/test_layout_opencl
#adb push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_conv_opencl /data/local/tmp/opencl/test_conv_opencl

#adb push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_reshape_opencl /data/local/tmp/opencl/test_reshape_opencl

adb push build.self.lite.android.armv7.clang.opencl/lite/kernels/opencl/${testname} /data/local/tmp/opencl/${testname}
# adb shell chmod +x /data/local/tmp/opencl/${testname}
adb shell "export GLOG_v=0; /data/local/tmp/opencl/${testname}"
