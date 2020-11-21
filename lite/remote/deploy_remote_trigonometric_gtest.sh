#!/usr/bin/env zsh
set -o errexit

#build_dir=build.self.withcv.lite.android.armv7.gcc.opencl
#build_dir=build.lite.android.armv7.clang.opencl
build_dir=build.self.lite.android.armv7.clang.opencl
testname="test_trigonometric_image_opencl"
# model_dir="/data/local/tmp/opencl/models/lens_yolonano_fluid_20200319"
pwd
scp Ubuntu_home:/workspace/Paddle-Lite/${build_dir}/lite/kernels/opencl/${testname} ./

adb push ./${testname} /data/local/tmp/opencl/${testname}

cmd="export GLOG_v=0; /data/local/tmp/opencl/${testname}"
echo ${cmd}
adb shell ${cmd}
