set -o errexit
IN_MODEL=$1
OUT_MODEL=$2

#IN_MODEL=../paddle-mobile-tools/models/m2fm_v18_18_9_grid_sampler/saved-20200316-125605/
#IN_MODEL=../paddle-mobile-tools/models/caffe2pd/caffe2pd_mobilenetv1/
#IN_MODEL=../paddle-mobile-tools/models/Lens_MnasNet/saved-20200214-094918/
#OUT_MODEL=a

#export GLOG_v=0
/data/opt_2.6.0 \
    --model_dir=${IN_MODEL} \
    --optimize_out_type=naive_buffer \
    --optimize_out=${OUT_MODEL}"_arm" \
    --valid_targets=arm #arm#opencl
#    --prefer_int8_kernel=(true|false) \
#    --record_tailoring_info =(true|false)

/data/opt_2.6.0 \
    --model_dir=${IN_MODEL} \
    --optimize_out_type=naive_buffer \
    --optimize_out=${OUT_MODEL}"_opencl" \
    --valid_targets=opencl