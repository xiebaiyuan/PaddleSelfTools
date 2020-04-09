#!/usr/bin/env zsh
set -o errexit

#build_dir=build.self.withcv.lite.android.armv7.gcc.opencl
#build_dir=build.lite.android.armv7.clang.opencl
build_dir=build.lite.android.armv7.clang.opencl
pwd
scp ubuntu_home:/workspace/Paddle-Lite/${build_dir}/lite/api/libpaddle_light_api_shared.so ./

cp ./libpaddle_light_api_shared.so ./lens_sdk/src/main/cpp/Thirdparty/mml/GPU/armeabi-v7a/libpaddle_light_api_shared.so
cp ./libpaddle_light_api_shared.so ./lens_sdk/src/main/jniLibs/armeabi-v7a/libpaddle_light_api_shared.so

echo "build_dir : $build_dir"

pwd

#./gradlew clean
./gradlew installDebug
