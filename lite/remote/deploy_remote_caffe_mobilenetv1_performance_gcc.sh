#!/usr/bin/env zsh
set -o errexit
build_dir=build.lite.android.armv7.gcc.opencl
pwd
scp Ubuntu_home:/workspace/Paddle-Lite/${build_dir}/inference_lite_lib.android.armv7.opencl/demo/cxx/mobile_light/mobilenetv1_light_api ./
adb push ./mobilenetv1_light_api /data/local/tmp/opencl/mobilenetv1_light_api
model_name="/data/local/tmp/opencl/benchmodels26/caffe2pd_mobilenetv1_opencl_fp32_opt_releasev2.6_b8234efb_20200423.nb"
cmd="export GLOG_v=0;cd /data/local/tmp/opencl/; export LD_LIBRARY_PATH=.; /data/local/tmp/opencl/mobilenetv1_light_api ${model_name}  1 3 224 224 1000 20 0"
echo ${cmd}
adb shell ${cmd}
