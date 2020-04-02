set -o errexit
cd /workspace/Paddle-Lite/build.self.lite.android.armv7.gcc.opencl
make publish_inference -j6
adb push
cd -
cd /workspace/Paddle-Lite/build.self.lite.android.armv7.gcc.opencl/inference_lite_lib.android.armv7.opencl/demo/cxx/mobile_light
make mobilenetv1_light_api -j6
cd -

adb push /workspace/Paddle-Lite/build.self.lite.android.armv7.gcc.opencl/inference_lite_lib.android.armv7.opencl/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/opencl
adb push /workspace/Paddle-Lite/build.self.lite.android.armv7.gcc.opencl/inference_lite_lib.android.armv7.opencl/demo/cxx/mobile_light/mobilenetv1_light_api /data/local/tmp/opencl
adb push /data/coremodels/performancemodelv3/performancemodelv3_opencl.nb /data/local/tmp/opencl

cp /workspace/Paddle-Lite/build.self.lite.android.armv7.gcc.opencl/inference_lite_lib.android.armv7.opencl/cxx/lib/libpaddle_light_api_shared.so /share
