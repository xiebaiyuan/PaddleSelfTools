#!/usr/bin/env zsh
set -o errexit

#build_dir=build.self.withcv.lite.android.armv7.gcc.opencl
#build_dir=build.lite.android.armv7.clang.opencl
build_dir=build.self.lite.android.armv7.clang.opencl
testname="test_mobilenetv1"
model_dir="/data/local/tmp/opencl/models/5_opencl_models_opt_dev_a50a8bea_20200319/lens_yolonano_fluid_20200319"
pwd
scp Ubuntu_home:/workspace/Paddle-Lite/${build_dir}/lite/api/${testname} ./

adb push ./${testname} /data/local/tmp/opencl/${testname}

cmd="export GLOG_v=0; /data/local/tmp/opencl/${testname} --model_dir=${model_dir} -N=1 -C=3 -H=416 -W=416"
echo ${cmd}
adb shell ${cmd}
