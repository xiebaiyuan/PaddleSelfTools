set -ex
echo '关闭omp.'
sed -i 's/LITE_WITH_OPENMP "Enable OpenMP in lite framework" ON/LITE_WITH_OPENMP "Enable OpenMP in lite framework" OFF/' CMakeLists.txt
cat CMakeLists.txt | grep "Enable OpenMP"
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

echo '打开omp'
sed -i 's/LITE_WITH_OPENMP "Enable OpenMP in lite framework" OFF/LITE_WITH_OPENMP "Enable OpenMP in lite framework" ON/' CMakeLists.txt
cat CMakeLists.txt | grep "Enable OpenMP"

echo '打包发布'
./PaddleSelfTools/release/build_lite_release_zip.sh
