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
    python3 $root_dir/lite/tools/cmake_tools/gen_opencl_code.py $OPENCL_KERNELS_PATH $GEN_CODE_PATH_OPENCL/opencl_kernels_source.cc
}
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


with_cmake=false
with_make=true
with_push=false

testname="paddle_light_api_shared"
build_dir=build.lite.android.armv7.clang.opencl

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
echo "build_dir : $build_dir"

pwd
if [[ "$with_cmake" == "true" ]]; then
    cd PaddleMobileTools
    ./ci_build_cmake.sh
    cd -
fi

if [[ "$with_make" == "true" ]]; then
    prepare_opencl_source_code $(pwd)

    cd ${build_dir}
    make $testname -j$6
    cd -
    # cd ${build_dir}/inference_lite_lib.android.armv7.opencl/cxx/lib
    cd ${build_dir}/lite/api
    adb -s ${devicename}  push libpaddle_light_api_shared.so /data/local/tmp/opencl
    cp libpaddle_light_api_shared.so /share
    cd -
    cd ${build_dir}/inference_lite_lib.android.armv7.opencl/demo/cxx/mobile_light
    make -j6

    adb -s ${devicename}  push mobilenetv1_light_api /data/local/tmp/opencl
    cd -
fi

if [[ "$with_push" == "true" ]]; then
    # 在/data/local/tmp目录下创建OpenCL文件目录
    # adb -s ${devicename}  shell mkdir -p /data/local/tmp/opencl
    # adb -s ${devicename}  shell mkdir -p /data/local/tmp/opencl/cl_kernel/buffer
    # adb -s ${devicename}  shell mkdir -p /data/local/tmp/opencl/cl_kernel/image

    # # 将OpenCL的kernels文件推送到/data/local/tmp/opencl目录下
    # adb -s ${devicename}  push lite/backends/opencl/cl_kernel/cl_common.h /data/local/tmp/opencl/cl_kernel/
    # adb -s ${devicename}  push lite/backends/opencl/cl_kernel/buffer/* /data/local/tmp/opencl/cl_kernel/buffer/
    # adb -s ${devicename}  push lite/backends/opencl/cl_kernel/image/* /data/local/tmp/opencl/cl_kernel/image/

    adb -s ${devicename}  shell mkdir -p ${model_dir}
    # adb -s ${devicename}  push ${input_dir}${input} /data/local/tmp/opencl/${input}
    # adb -s ${devicename}  push ${output_dir}${output} /data/local/tmp/opencl/${output}
    # adb -s ${devicename}  push ${source_model_dir}/* ${model_dir}
fi

# adb -s ${devicename}  shell mkdir -p /data/local/tmp/opencl/mobilenet_v1
#adb -s ${devicename}  push build.lite.android.armv8.gcc.opencl/third_party/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1/

# 将OpenCL单元测试程序test_mobilenetv1，推送到/data/local/tmp/opencl目录下
#adb -s ${devicename}  push build.lite.android.armv8.gcc.opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl

#adb -s ${devicename}  push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_layout_opencl /data/local/tmp/opencl/test_layout_opencl
#adb -s ${devicename}  push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_conv_opencl /data/local/tmp/opencl/test_conv_opencl

#adb -s ${devicename}  push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_reshape_opencl /data/local/tmp/opencl/test_reshape_opencl

# adb -s ${devicename}  push ${build_dir}/lite/kernels/opencl/${testname} /data/local/tmp/opencl/${testname}
# adb -s ${devicename}  shell chmod +x /data/local/tmp/opencl/mobilenetv1_light_api
# model_name="caffe2pd_mobilenetv1_opencl_opt_dev_b740c549_20200323.nb"
model_name="airank_mobilenet_v1_opencl.nb"
# model_name="lens_yolonano_opencl_dev_a18ca82e.nb"
# model_name="caffe2pd_mobilenetv2_opencl_opt_dev_b740c549_20200323.nb"
cmd="export Host2CLMode=2; export GLOG_v=0;cd /data/local/tmp/opencl/; export LD_LIBRARY_PATH=.; /data/local/tmp/opencl/mobilenetv1_light_api /data/local/tmp/opencl/models/${model_name} 1,3,224,224 10000 10 0 1 0 0"
echo ${cmd}
adb -s ${devicename}  shell ${cmd}
