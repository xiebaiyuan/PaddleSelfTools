src="https:\/\/github.com\/KhronosGroup\/OpenCL-Headers.git"
target="https:\/\/gitee.com\/xiebaiyuan\/OpenCL-Headers.git"
perl -pi -e "s/${src}/${target}/g" "cmake/external/opencl-headers.cmake"

cat cmake/external/opencl-headers.cmake | grep "GIT_REPOSITORY"



perl -pi -e "s/https:\/\/github.com\/KhronosGroup\/OpenCL-CLHPP.git/https:\/\/gitee.com\/xiebaiyuan\/OpenCL-CLHPP.git/g" "cmake/external/opencl-clhpp.cmake"
cat cmake/external/opencl-clhpp.cmake | grep "GIT_REPOSITORY"

perl -pi -e "s/https:\/\/github.com\/Shixiaowei02\/flatbuffers.git/https:\/\/gitee.com\/xiebaiyuan\/flatbuffers.git/g" "cmake/external/flatbuffers.cmake"
cat cmake/external/flatbuffers.cmake | grep "GIT_REPOSITORY"
