set -ex
# git fetch
# git merge

# set +ex
# read last_commit < 'last_commit.log'
# echo "line : $last_commit"

# current_commit=$(git rev-parse HEAD)
# if [ "$current_commit" = "$last_commit" ];then
# echo "[ 没有新的提交. 跳过...... ]"
# exit 0
# fi

# set -ex

export NDK_ROOT=/opt/android-ndk-r17c
# 删除上一次的构建产物
rm -rf build.*
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

# echo 'arm v8'
# ./lite/tools/build_android.sh \
# --arch=armv8 \
# --toolchain=clang \
# --android_stl=c++_static \
# --with_java=ON \
# --with_cv=ON \
# --with_log=OFF \
# --with_extra=ON \
# --with_opencl=OFF

# echo 'arm v7'
# ./lite/tools/build_android.sh \
# --arch=armv7 \
# --toolchain=clang \
# --android_stl=c++_static \
# --with_java=ON \
# --with_cv=ON \
# --with_log=OFF \
# --with_extra=ON \
# --with_opencl=OFF


# echo 'opencl v8'
# ./lite/tools/build_android.sh \
# --arch=armv8 \
# --toolchain=clang \
# --android_stl=c++_static \
# --with_java=ON \
# --with_cv=ON \
# --with_log=OFF \
# --with_extra=ON \
# --with_opencl=ON

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
echo '打包发布'
./PaddleSelfTools/release/build_lite_release_zip.sh

# echo '记录上次打包成功commit号'
# git rev-parse HEAD > last_commit.log
