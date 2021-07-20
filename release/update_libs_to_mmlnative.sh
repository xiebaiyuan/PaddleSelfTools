#!/bin/bash
set -e
WITH_COMMIT=OFF
COMMIT_MSG=""
function md5_sum {
        local path=$1
        echo -e "\033[32m"md5 in "$path""\033[0m"
        find "$path" -type f -print0 | xargs -0 md5sum
}

function update_libs {
    local mml_native_path=$1
    local lib_path=$2
    
    # tree $lib_path
    mml_native_libs_dst=$mml_native_path/mml_framework/src/main/jniLibs

    src_v7_cpu="$lib_path"/inference_lite_lib.android.armv7/cxx/lib/
    src_v8_cpu="$lib_path"/inference_lite_lib.android.armv8/cxx/lib/
    src_v7=$lib_path/inference_lite_lib.android.armv7.opencl/cxx/lib/
    src_v8=$lib_path/inference_lite_lib.android.armv8.opencl/cxx/lib/

    src_include_cl=$lib_path/inference_lite_lib.android.armv7.opencl/cxx/include/
    src_include_arm=$lib_path/inference_lite_lib.android.armv7/cxx/include/

    echo "                                     "
    md5_sum "$src_v7"
    md5_sum "$src_v7_cpu"

    md5_sum "$src_v8"
    md5_sum "$src_v8_cpu"
    echo "                                     "

    dst_v7="$mml_native_libs_dst"/armeabi-v7a/
    dst_v8="$mml_native_libs_dst"/arm64-v8a/

    dst_include_cl=$mml_native_path/mml_framework/src/main/cpp/mml_framework/paddle_lite_header_android_arm_opencl/
    dst_include_arm=$mml_native_path/mml_framework/src/main/cpp/mml_framework/paddle_lite_header_android_arm/

    ls "$mml_native_libs_dst"
    echo "$mml_native_path"
    echo "$lib_path"
    
    set +ex
    rm "$mml_native_libs_dst"/arm64-v8a/libpaddle*
    rm "$mml_native_libs_dst"/armeabi-v7a/libpaddle*
    rm "$dst_include_cl"/*
    rm "$dst_include_arm"/*
    set -ex


    # cp cpu v7
    cp "$src_v7_cpu"/libpaddle_light_api_shared.so "$dst_v7"/libpaddle_light_api_shared_cpu_only.so
    cp "$src_v7_cpu"/libpaddle_api_light_bundled.a "$dst_v7"/libpaddle_api_light_bundled_cpu_only.a

    # cp cpu v8
    cp "$src_v8_cpu"/libpaddle_light_api_shared.so "$dst_v8"/libpaddle_light_api_shared_cpu_only.so
    cp "$src_v8_cpu"/libpaddle_api_light_bundled.a "$dst_v8"/libpaddle_api_light_bundled_cpu_only.a
    

    # cp gpu v7
    cp "$src_v7"/* "$dst_v7"/
    # cp gpu v8
    cp "$src_v8"/* "$dst_v8"/

    md5_sum "$dst_v7"
    md5_sum "$dst_v8"

    cp "$src_include_cl"/* "$dst_include_cl"/
    cp "$src_include_arm"/* "$dst_include_arm"/


#    sed -i 's/readonly NUM_PROC=${LITE_BUILD_THREADS:-4}/readonly NUM_PROC=30 #${LITE_BUILD_THREADS:-4}/' ./lite/tools/build.sh

#    cat "$dst_include_cl"/paddle_image_preprocess.h
#    sed -i 's/\/lite\/api\/paddle_api.h/paddle_api.h/' "$dst_include_cl"/paddle_image_preprocess.h
#    sed -i 's/a/b/' "$dst_include_cl"/paddle_image_preprocess.h
#    sed -i 's/lite/api/paddle_api.h\paddle_api.h/' "$dst_include_cl"/paddle_image_preprocess.h
#    sed -i 's/#include \"lite/api/paddle_api.h"/#include \"paddle_api.h\"/' "$dst_include_arm"/paddle_image_preprocess.h

#    sed -i '' 's/paddle/apple/g' "$dst_include_cl"/paddle_image_preprocess.h
    sed -i '' 's/lite\/api\/paddle_api.h/paddle_api.h/g' "$dst_include_cl"/paddle_image_preprocess.h
    sed -i '' 's/lite\/api\/paddle_place.h/paddle_place.h/g' "$dst_include_cl"/paddle_image_preprocess.h

    sed -i '' 's/lite\/api\/paddle_api.h/paddle_api.h/g' "$dst_include_arm"/paddle_image_preprocess.h
    sed -i '' 's/lite\/api\/paddle_place.h/paddle_place.h/g' "$dst_include_arm"/paddle_image_preprocess.h
#    sed -i '' 's/paddle/apple/g' "$dst_include_cl"/paddle_image_preprocess.h

   if [ ${WITH_COMMIT} == "ON" ]; then
      cd "$mml_native_path"
      git fetch
      git status
      git add -u
#      git commit -m $COMMIT_MSG
      echo "$COMMIT_MSG"
      git commit -m "$COMMIT_MSG" || true
      git push origin HEAD:refs/for/master
   fi


}
 
 
 
function print_usage {
    set +x
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "./update_libs_to_mmlnative.sh"
    echo
    echo -e "optional argument:"
    echo -e "--mmlnative_path:"
    echo -e "--libs_path: "
    echo -e "--commit: "
    echo "----------------------------------------"
    echo
}
 
function main {
    if [ -z "$1" ]; then
        print_usage
        exit 1
    fi
 
    # Parse command line.
    for i in "$@"; do
        case $i in
            --mmlnative_path=*)
                MML_NATIVE_PATH="${i#*=}"
                shift
                ;;
            --libs_path=*)
                LIBS_PATH="${i#*=}"
                shift
                ;;
            --commit=*)
                WITH_COMMIT="${i#*=}"
                shift
                ;;
            --commit_msg=*)
                COMMIT_MSG="${i#*=}"
                shift
                ;;
            update_libs)
                update_libs $MML_NATIVE_PATH $LIBS_PATH
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
}
 
main $@