#!/usr/bin/env bash
set -o errexit
with_cmake=false
with_make=true
with_push=true
source_model_dir="/data/coremodels/performancemodelv3/split_model"
# source_model_dir="/data/self_model_gen/mnasnet_self_saved-20200417-170543"
# source_model_dir="/data/self_model_gen/mnasnet_self_saved-20200417-170927"
model_dir="/data/local/tmp/opencl/models/pfj/"
# model_dir="/data/local/tmp/opencl/models/lens_mnasnet_caffe_part/"
# source_model_dir="/data/MnasNet-caffe/pd_model/inference_model"
# model_dir="/data/local/tmp/opencl/models/lens_mnasnet_self/"
input="image"
output="save_infer_model_scale_0"
# source_model_dir="/data/MnasNet-caffe/pd_model/model_with_code/lens_mnasnet_part"

# model_dir="/data/local/tmp/opencl/models/aaaaaaaa"
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
    ./ci_build_cmake.sh
fi

if [[ "$with_make" == "true" ]]; then
    python3 ./lite/tools/cmake_tools/gen_opencl_code.py ./lite/backends/opencl/cl_kernel ./lite/backends/opencl/opencl_kernels_source.cc
    cd build.self.lite.android.armv7.clang.opencl

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
    # adb push ${input_dir}${input} /data/local/tmp/opencl/${input}
    # adb push ${output_dir}${output} /data/local/tmp/opencl/${output}
    adb push ${source_model_dir}/* ${model_dir}
fi

# adb shell mkdir -p /data/local/tmp/opencl/mobilenet_v1
#adb push build.lite.android.armv8.gcc.opencl/third_party/install/mobilenet_v1/* /data/local/tmp/opencl/mobilenet_v1/

# 将OpenCL单元测试程序test_mobilenetv1，推送到/data/local/tmp/opencl目录下
#adb push build.lite.android.armv8.gcc.opencl/lite/api/test_mobilenetv1 /data/local/tmp/opencl

#adb push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_layout_opencl /data/local/tmp/opencl/test_layout_opencl
#adb push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_conv_opencl /data/local/tmp/opencl/test_conv_opencl

#adb push build.lite.android.armv8.gcc.opencl/lite/kernels/opencl/test_reshape_opencl /data/local/tmp/opencl/test_reshape_opencl

adb push build.self.lite.android.armv7.clang.opencl/lite/api/${testname} /data/local/tmp/opencl/${testname}
# adb shell chmod +x /data/local/tmp/opencl/${testname}
cmd="export GLOG_v=0; /data/local/tmp/opencl/${testname} --model_dir=${model_dir} -N=1 -C=3 -H=224 -W=224 --optimized_model=/data/local/tmp/pfj"
echo ${cmd}
adb shell ${cmd}
adb pull /data/local/tmp/mnasetnet_caffee.nb ./
md5sum pfj.nb
cp pfj.nb ${source_model_dir}
