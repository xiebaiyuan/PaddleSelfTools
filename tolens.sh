#!/usr/bin/env bash
lens_dir="/Users/xiebaiyuan/PaddleProject/baidu/mms-android/lens"
cp "../build/package/libpaddle-mobile-v7a-cpu.so" ""${lens_dir}
v8=false
cpu=false


cp ../build/package/libpaddle-mobile-v7a-cpu.so ${lens_dir}/lens_sdk/src/main/cpp/Thirdparty/PaddleMobile/CPU/armeabi-v7a/libpaddle-mobile.so
cp ../build/package/libpaddle-mobile-v7a-gpu.so ${lens_dir}/lens_sdk/src/main/cpp/Thirdparty/PaddleMobile/GPU/armeabi-v7a/libpaddle-mobile.so

if [[ "$cpu" == "true" ]]; then
      echo $cpu

  cp ../build/package/libpaddle-mobile-v7a-cpu.so ${lens_dir}/lens_sdk/src/main/jniLibs/armeabi-v7a/libpaddle-mobile.so
else
  cp ../build/package/libpaddle-mobile-v7a-gpu.so ${lens_dir}/lens_sdk/src/main/jniLibs/armeabi-v7a/libpaddle-mobile.so
fi


#if [ "$v8" == true ]; then
  cp ../build/package/libpaddle-mobile-v8a-cpu.so ${lens_dir}/lens_sdk/src/main/cpp/Thirdparty/PaddleMobile/CPU/arm64-v8a/libpaddle-mobile.so
  cp ../build/package/libpaddle-mobile-v8a-gpu.so ${lens_dir}/lens_sdk/src/main/cpp/Thirdparty/PaddleMobile/GPU/arm64-v8a/libpaddle-mobile.so
  if [[ "$cpu" == "true" ]]; then
    echo $cpu
      cp ../build/package/libpaddle-mobile-v8a-cpu.so ${lens_dir}/lens_sdk/src/main/jniLibs/arm64-v8a/libpaddle-mobile.so
  else
      cp ../build/package/libpaddle-mobile-v8a-gpu.so ${lens_dir}/lens_sdk/src/main/jniLibs/arm64-v8a/libpaddle-mobile.so

  fi
#fi



echo "done!"