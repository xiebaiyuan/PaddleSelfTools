#!/usr/bin/env bash
set -o errexit
with_cmake=false
with_make=false
with_push=true

#input_dir="/data/coremodels/lens_nanoyolo/feeds/"
#output_dir="/data/coremodels/lens_nanoyolo/outputs/"
#input="image"
#output="save_infer_model_scale_0"
source_model_dir="/Users/xiebaiyuan/workspace/github/Paddle-Lite-Vulkan-Dev/mobilenetv1"
model_dir="/data/local/tmp/vulkan/models/mobilenetv1/"
testname="test_mobilenetv1"

echo "with cmake : $with_cmake"
echo "with_make : $with_make"
echo "with_push : $with_push"
echo "input_dir : $input_dir"
echo "output_dir : $output_dir"
echo "input : $input"
echo "output : $output"
echo "source_model_dir : $source_model_dir"
echo "model_dir : $model_dir"
echo "testname : $testname"

pwd
if [[ "$with_cmake" == "true" ]]; then
    ./ci_build_cmake_vulkan.sh
fi

if [[ "$with_make" == "true" ]]; then
#    python ./lite/tools/cmake_tools/gen_opencl_code.py ./lite/backends/opencl/cl_kernel ./lite/backends/opencl/opencl_kernels_source.cc
    cd build.lite.android.armv8.clang.vulkan
    make $testname -j$4
    cd -
fi

if [[ "$with_push" == "true" ]]; then
    #  do not need upload .cl
    # # 在/data/local/tmp目录下创建OpenCL文件目录
    # adb shell mkdir -p /data/local/tmp/opencl
    # adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/buffer
    # adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/image

    # # 将OpenCL的kernels文件推送到/data/local/tmp/opencl目录下
    # adb push lite/backends/opencl/cl_kernel/cl_common.h /data/local/tmp/opencl/cl_kernel/
    # adb push lite/backends/opencl/cl_kernel/buffer/* /data/local/tmp/opencl/cl_kernel/buffer/
    # adb push lite/backends/opencl/cl_kernel/image/* /data/local/tmp/opencl/cl_kernel/image/

    adb shell mkdir -p ${model_dir}
    adb push ${source_model_dir}/* ${model_dir}
fi

# adb shell mkdir -p /data/local/tmp/opencl/mobilenet_v1
#adb push build.lite.android.armv8.clang.opencl/third_party/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1/

# 将OpenCL单元测试程序test_mobilenetv1，推送到/data/local/tmp/opencl目录下
#adb push build.lite.android.armv8.clang.opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl

#adb push build.lite.android.armv8.clang.opencl/lite/kernels/opencl/test_layout_opencl /data/local/tmp/opencl/test_layout_opencl
#adb push build.lite.android.armv8.clang.opencl/lite/kernels/opencl/test_conv_opencl /data/local/tmp/opencl/test_conv_opencl

#adb push build.lite.android.armv8.clang.opencl/lite/kernels/opencl/test_reshape_opencl /data/local/tmp/opencl/test_reshape_opencl

adb push build.lite.android.armv8.clang.vulkan/lite/api/test/${testname} /data/local/tmp/opencl/${testname}
adb shell chmod +x /data/local/tmp/opencl/${testname}
cmd="export GLOG_v=1; /data/local/tmp/opencl/${testname} --model_dir=${model_dir} -N=1 -C=3 -H=224 -W=224 -warmup=100 -repeats=10"
echo ${cmd}
adb shell ${cmd}
