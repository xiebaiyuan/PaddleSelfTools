set -ex

export NDK_ROOT=/opt/android-ndk-r17c
# 删除上一次的构建产物
rm -rf build.*
# 删除上一次CMake自动生成的.h文件
rm ./lite/api/paddle_use_kernels.h || true
rm ./lite/api/paddle_use_ops.h || true

echo "NDK_ROOT: ${NDK_ROOT}"
echo "ANDROID_NDK: ${ANDROID_NDK}"
echo '关闭omp.'
sed -i 's/lite_option(LITE_WITH_OPENMP    "Enable OpenMP in lite framework" ON)/lite_option(LITE_WITH_OPENMP    "Enable OpenMP in lite framework" OFF)/' CMakeLists.txt
cat CMakeLists.txt | grep "LITE_WITH_OPENMP"

echo "修改线程数量"
sed -i 's/readonly NUM_PROC=${LITE_BUILD_THREADS:-4}/readonly NUM_PROC=30 #${LITE_BUILD_THREADS:-4}/' ./lite/tools/build.sh
sed -i 's/readonly NUM_PROC=${LITE_BUILD_THREADS:-4}/readonly NUM_PROC=30 #${LITE_BUILD_THREADS:-4}/' ./lite/tools/build_android.sh
sed -i 's/readonly NUM_PROC=${LITE_BUILD_THREADS:-4}/readonly NUM_PROC=30 #${LITE_BUILD_THREADS:-4}/' ./lite/tools/ci_build.sh


echo 'opencl v8'
./lite/tools/build_android.sh \
--arch=armv8 \
--toolchain=clang \
--android_stl=c++_static \
--with_java=ON \
--with_exception=ON \
--with_static_lib=ON \
--with_cv=ON \
--with_log=ON \
--with_extra=ON \
--with_opencl=ON

echo 'opencl v7'
./lite/tools/build_android.sh \
--arch=armv7 \
--toolchain=clang \
--android_stl=c++_static \
--with_java=ON \
--with_cv=ON \
--with_exception=ON \
--with_static_lib=ON \
--with_log=ON \
--with_extra=ON \
--with_opencl=ON


