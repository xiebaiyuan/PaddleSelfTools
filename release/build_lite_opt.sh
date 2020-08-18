set -ex
export NDK_ROOT=/opt/android-ndk-r17c


echo 'BUILDING OPT ......  '
./lite/tools/build.sh build_optimize_tool
