#!/usr/bin/env bash
libprefix=libpaddle-mobile-lens

lens_dir="/Users/xiebaiyuan/PaddleProject/baidu/mms-android/lens"
#cp "../build/package/libpaddle-mobile-v7a-cpu.so" ""${lens_dir}
v8=true
cpu=false

cp ../build/package/${libprefix}-v7a-cpu.so ${lens_dir}/lens_sdk/src/main/cpp/Thirdparty/PaddleMobile/CPU/armeabi-v7a/${libprefix}.so
cp ../build/package/${libprefix}-v7a-gpu.so ${lens_dir}/lens_sdk/src/main/cpp/Thirdparty/PaddleMobile/GPU/armeabi-v7a/${libprefix}.so

if [[ "$cpu" == "true" ]]; then
      echo $cpu
  cp ../build/package/${libprefix}-v7a-cpu.so ${lens_dir}/lens_sdk/src/main/jniLibs/armeabi-v7a/${libprefix}.so
else
  cp ../build/package/${libprefix}-v7a-gpu.so ${lens_dir}/lens_sdk/src/main/jniLibs/armeabi-v7a/${libprefix}.so
fi


if [[ "$v8" == "true" ]]; then
  cp ../build/package/${libprefix}-v8a-cpu.so ${lens_dir}/lens_sdk/src/main/cpp/Thirdparty/PaddleMobile/CPU/arm64-v8a/${libprefix}.so
  cp ../build/package/${libprefix}-v8a-gpu.so ${lens_dir}/lens_sdk/src/main/cpp/Thirdparty/PaddleMobile/GPU/arm64-v8a/${libprefix}.so
  if [[ "$cpu" == "true" ]]; then
    echo $cpu
    cp ../build/package/${libprefix}-v8a-cpu.so ${lens_dir}/lens_sdk/src/main/jniLibs/arm64-v8a/${libprefix}.so
  else
    cp ../build/package/${libprefix}-v8a-gpu.so ${lens_dir}/lens_sdk/src/main/jniLibs/arm64-v8a/${libprefix}.so
  fi
fi



echo "done!"