#!/usr/bin/env bash
set -o errexit
with_cmake=false
with_make=true
with_push=true

input_dir="/data/coremodels/lens_nanoyolo/feeds/"
output_dir="/data/coremodels/lens_nanoyolo/outputs/"
input="image"
output="save_infer_model_scale_0"
source_model_dir="/data/standard_models/paddles/airank/airank/paddlelite/f32/MobileNetV1_infer/MobileNetV1_infer-saved-20210513-183038"
model_dir="/data/local/tmp/opencl/models/airank/mobilenetv1"
testname="test_mobilenetv1"

# vivo neon 3
# devicename="516efcd5" 

# v10 
# devicename="RKK0218104002556"

# 845
# devicename="f5caa946"

# mate30
 devicename="7HX5T19929012679"

# f5caa946	
# 7HX5T19929012679	device
# 516efcd5	device
# RKK0218104002556	device

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
    ./ci_build_cmake.sh
fi

if [[ "$with_make" == "true" ]]; then
    python ./lite/tools/cmake_tools/gen_opencl_code.py ./lite/backends/opencl/cl_kernel ./lite/backends/opencl/opencl_kernels_source.cc
    cd build.self.lite.android.armv7.clang.opencl

    make $testname -j$4
    cd -
fi

if [[ "$with_push" == "true" ]]; then
    #  do not need upload .cl
    # # 在/data/local/tmp目录下创建OpenCL文件目录
    # adb -s ${devicename} shell mkdir -p /data/local/tmp/opencl
    # adb -s ${devicename} shell mkdir -p /data/local/tmp/opencl/cl_kernel/buffer
    # adb -s ${devicename} shell mkdir -p /data/local/tmp/opencl/cl_kernel/image

    # # 将OpenCL的kernels文件推送到/data/local/tmp/opencl目录下
    # adb -s ${devicename} push lite/backends/opencl/cl_kernel/cl_common.h /data/local/tmp/opencl/cl_kernel/
    # adb -s ${devicename} push lite/backends/opencl/cl_kernel/buffer/* /data/local/tmp/opencl/cl_kernel/buffer/
    # adb -s ${devicename} push lite/backends/opencl/cl_kernel/image/* /data/local/tmp/opencl/cl_kernel/image/

    adb -s ${devicename} shell mkdir -p ${model_dir}
    adb -s ${devicename} push ${source_model_dir}/* ${model_dir}
fi

# adb -s ${devicename} shell mkdir -p /data/local/tmp/opencl/mobilenet_v1
#adb -s ${devicename} push build.lite.android.armv8.clang.opencl/third_party/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1/

# 将OpenCL单元测试程序test_mobilenetv1，推送到/data/local/tmp/opencl目录下
#adb -s ${devicename} push build.lite.android.armv8.clang.opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl

#adb -s ${devicename} push build.lite.android.armv8.clang.opencl/lite/kernels/opencl/test_layout_opencl /data/local/tmp/opencl/test_layout_opencl
#adb -s ${devicename} push build.lite.android.armv8.clang.opencl/lite/kernels/opencl/test_conv_opencl /data/local/tmp/opencl/test_conv_opencl

#adb -s ${devicename} push build.lite.android.armv8.clang.opencl/lite/kernels/opencl/test_reshape_opencl /data/local/tmp/opencl/test_reshape_opencl

adb -s ${devicename} push build.self.lite.android.armv7.clang.opencl/lite/api/${testname} /data/local/tmp/opencl/${testname}
adb -s ${devicename} shell chmod +x /data/local/tmp/opencl/${testname}
cmd="export Host2CLMode=1; export GLOG_v=0; /data/local/tmp/opencl/${testname} --model_dir=${model_dir} -N=1 -C=3 -H=224 -W=224 -warmup=10 -repeats=10"
echo ${cmd}
adb -s ${devicename} shell ${cmd}


cmd="export Host2CLMode=0; export GLOG_v=0; /data/local/tmp/opencl/${testname} --model_dir=${model_dir} -N=1 -C=3 -H=224 -W=224 -warmup=10 -repeats=10"
echo ${cmd}
adb -s ${devicename} shell ${cmd}
