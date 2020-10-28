set -ex
if [ ! -d paddle_dev_develop  ];then
  mkdir paddle_dev_develop
else
  echo paddle_dev_develop exist
fi

cd paddle_dev_develop

current_git_branch_latest_id=`git rev-parse HEAD`
echo "${current_git_branch_latest_id}"
current_git_branch_latest_short_id=`git rev-parse --short HEAD`
echo "${current_git_branch_latest_short_id}"

time=$(date "+%Y_%m%d_%H%M_%S")
echo "${time}"

package_dir="paddle_lite_buildtime_${time}_commitid_${current_git_branch_latest_short_id}"
echo "${package_dir}"
mkdir "${package_dir}"

cd "${package_dir}"
pwd
mkdir paddlelite_cpu_only_v7
cp -r ../../build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/* ./paddlelite_cpu_only_v7
mkdir paddlelite_cpu_only_v8
cp -r ../../build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/* ./paddlelite_cpu_only_v8
mkdir paddlelite_v7
cp -r ../../build.lite.android.armv7.clang.opencl/inference_lite_lib.android.armv7.opencl/cxx/* ./paddlelite_v7
mkdir paddlelite_v8
cp -r ../../build.lite.android.armv8.clang.opencl/inference_lite_lib.android.armv8.opencl/cxx/* ./paddlelite_v8
cp -r ../../build.opt/lite/api/opt ./
mkdir mml_deps
cd mml_deps

mkdir armeabi-v7a
cp ../paddlelite_cpu_only_v7/lib/libpaddle_api_light_bundled.a ./armeabi-v7a/libpaddle_api_light_bundled_cpu_only.a
cp ../paddlelite_cpu_only_v7/lib/libpaddle_light_api_shared.so ./armeabi-v7a/libpaddle_light_api_shared_cpu_only.so
cp ../paddlelite_v7/lib/libpaddle_api_light_bundled.a ./armeabi-v7a/libpaddle_api_light_bundled.a
cp ../paddlelite_v7/lib/libpaddle_light_api_shared.so ./armeabi-v7a/libpaddle_light_api_shared.so
mkdir arm64-v8a
cp ../paddlelite_cpu_only_v8/lib/libpaddle_api_light_bundled.a ./arm64-v8a/libpaddle_api_light_bundled_cpu_only.a
cp ../paddlelite_cpu_only_v8/lib/libpaddle_light_api_shared.so ./arm64-v8a/libpaddle_light_api_shared_cpu_only.so
cp ../paddlelite_v8/lib/libpaddle_api_light_bundled.a ./arm64-v8a/libpaddle_api_light_bundled.a
cp ../paddlelite_v8/lib/libpaddle_light_api_shared.so ./arm64-v8a/libpaddle_light_api_shared.so
cp -r ../paddlelite_cpu_only_v7/include/ ./
cd ..
cd ..

pwd


zip -r "${package_dir}.zip" "./${package_dir}"
cd  ..
