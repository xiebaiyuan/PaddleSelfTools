#!/bin/bash

set -ex
build_v8=false
build_armonly=false

if [ $# -lt 2 ]
then 
    echo "build develop"
else
    if [ $1 -eq 1 ]
    then
        build_v8=true
        echo "with v8"
    else
        echo "build v7 only"
    fi

    if [ $2 -eq 1 ]
    then
        build_armonly=true
        echo "with arm only"
    else
        echo "only combined pack"
    fi
fi




export NDK_ROOT=/opt/android-ndk-r17c
# 删除上一次的构建产物
#rm -rf build.*
# 删除上一次CMake自动生成的.h文件
rm ./lite/api/paddle_use_kernels.h || true
rm ./lite/api/paddle_use_ops.h || true

echo "NDK_ROOT: ${NDK_ROOT}"
echo "ANDROID_NDK: ${ANDROID_NDK}"
echo '关闭omp.'
sed -i 's/LITE_WITH_OPENMP "Enable OpenMP in lite framework" ON/LITE_WITH_OPENMP "Enable OpenMP in lite framework" OFF/' CMakeLists.txt
cat CMakeLists.txt | grep "Enable OpenMP"

echo "修改线程数量"
sed -i 's/readonly NUM_PROC=${LITE_BUILD_THREADS:-4}/readonly NUM_PROC=30 #${LITE_BUILD_THREADS:-4}/' ./lite/tools/build.sh
sed -i 's/readonly NUM_PROC=${LITE_BUILD_THREADS:-4}/readonly NUM_PROC=30 #${LITE_BUILD_THREADS:-4}/' ./lite/tools/build_android.sh
sed -i 's/readonly NUM_PROC=${LITE_BUILD_THREADS:-4}/readonly NUM_PROC=30 #${LITE_BUILD_THREADS:-4}/' ./lite/tools/ci_build.sh


if [ "$build_armonly" = "true" ]
then
echo 'arm v8'
./lite/tools/build_android.sh \
--arch=armv8 \
--toolchain=clang \
--android_stl=c++_static \
--with_java=ON \
--with_cv=ON \
--with_log=OFF \
--with_extra=ON \
--with_opencl=OFF

echo 'arm v7'
./lite/tools/build_android.sh \
--arch=armv7 \
--toolchain=clang \
--android_stl=c++_static \
--with_java=ON \
--with_cv=ON \
--with_log=OFF \
--with_extra=ON \
--with_opencl=OFF
fi


if [ "$build_v8" = "true"]
then
echo 'opencl v8'
./lite/tools/build_android.sh \
--arch=armv8 \
--toolchain=clang \
--android_stl=c++_static \
--with_java=ON \
--with_cv=ON \
--with_log=OFF \
--with_extra=ON \
--with_opencl=ON
fi


echo 'opencl v7'
./lite/tools/build_android.sh \
--arch=armv7 \
--toolchain=clang \
--android_stl=c++_static \
--with_java=ON \
--with_cv=ON \
--with_log=OFF \
--with_extra=ON \
--with_opencl=ON

# ./PaddleSelfTools/release/build_lite_opt.sh


echo '打开omp'
sed -i 's/LITE_WITH_OPENMP "Enable OpenMP in lite framework" OFF/LITE_WITH_OPENMP "Enable OpenMP in lite framework" ON/' CMakeLists.txt
cat CMakeLists.txt | grep "Enable OpenMP"

git checkout ./lite/tools/build.sh
git checkout ./lite/tools/build_android.sh
git checkout ./lite/tools/ci_build.sh



