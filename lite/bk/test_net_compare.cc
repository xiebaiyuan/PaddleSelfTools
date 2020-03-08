// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/kernels/opencl/test_helper.h"
#include "lite/tests/utils/tensor_utils.h"
DEFINE_string(optimized_model, "", "optimized_model");
DEFINE_int32(N, 1, "input_batch");
DEFINE_int32(C, 3, "input_channel");
DEFINE_int32(H, 416, "input_height");
DEFINE_int32(W, 416, "input_width");
DEFINE_string(input_file, "image", "input_file_path");
DEFINE_string(output_file, "save_infer_model_scale_0", "output_file_path");

DEFINE_bool(checkscript, false, "checkscript");
DEFINE_bool(check_nb, false, "check_nb");

DEFINE_bool(is_sample_step, false, "is_sample_step");
DEFINE_int32(sample_step, 1, "sample_step");
DEFINE_int32(sample_num, 20, "sample_num");
DEFINE_bool(check_shape, false, "check_shape");

namespace paddle {
namespace lite {
std::vector<std::string> exclude_names = {
    //"image",
    // "conv2d_0.tmp_0",
    // "conv2d_0.tmp_1",
    // "h2_02_upsample.tmp_0",
    // "concat_0.tmp_0",
    // "concat_1.tmp_0",
    // "concat_2.tmp_0",
    // "concat_3.tmp_0",
    // "h1_02_upsample.tmp_0",
    // "save_infer_model/scale_0"

};

// performancev3
std::vector<std::string> tensor_names_performancev3 = {"data",
                                                       "conv1.tmp_0",
                                                       "conv1_bn.tmp_2",
                                                       "relu1.tmp_0",
                                                       "conv2_1_dw.tmp_0",
                                                       "conv2_1_dw_bn.tmp_2",
                                                       "relu2_1_dw.tmp_0",
                                                       "conv2_1_sep.tmp_0",
                                                       "conv2_1_sep_bn.tmp_2",
                                                       "relu2_1_sep.tmp_0",
                                                       "conv2_2_dw.tmp_0",
                                                       "conv2_2_dw_bn.tmp_2",
                                                       "relu2_2_dw.tmp_0",
                                                       "conv2_2_sep.tmp_0",
                                                       "conv2_2_sep_bn.tmp_2",
                                                       "relu2_2_sep.tmp_0"};
// mnasnet
std::vector<std::string> tensor_names_mnasnet = {
    "image",
    "conv_stem.tmp_0",
    "conv_stem.tmp_1",
    "relu6_0.tmp_0",
    "sepconv.conv_dw.tmp_0",
    "sepconv.conv_dw.tmp_1",
    "relu6_1.tmp_0",
    "sepconv.conv_pw.tmp_0",
    "sepconv.conv_pw.tmp_1",
    "blocks.1.0.conv_pw.tmp_0",
    "blocks.1.0.conv_pw.tmp_1",
    "relu6_2.tmp_0",
    "blocks.1.0.conv_dw.tmp_0",
    "blocks.1.0.conv_dw.tmp_1",
    "relu6_3.tmp_0",
    "blocks.1.0.conv_pwl.tmp_0",
    "blocks.1.0.conv_pwl.tmp_1",
    "blocks.1.1.conv_pw.tmp_0",
    "blocks.1.1.conv_pw.tmp_1",
    "relu6_4.tmp_0",
    "blocks.1.1.conv_dw.tmp_0",
    "blocks.1.1.conv_dw.tmp_1",
    "relu6_5.tmp_0",
    "blocks.1.1.conv_pwl.tmp_0",
    "blocks.1.1.conv_pwl.tmp_1",
    "elementwise_add_0",
    "blocks.2.0.conv_pw.tmp_0",
    "blocks.2.0.conv_pw.tmp_1",
    "relu6_6.tmp_0",
    "blocks.2.0.conv_dw.tmp_0",
    "blocks.2.0.conv_dw.tmp_1",
    "relu6_7.tmp_0",
    "pool2d_0.tmp_0",
    "blocks.2.0.se.conv_reduce.tmp_0",
    "blocks.2.0.se.conv_reduce.tmp_1",
    "blocks.2.0.se.conv_reduce.tmp_2",
    "blocks.2.0.se.conv_expand.tmp_0",
    "blocks.2.0.se.conv_expand.tmp_1",
    "blocks.2.0.se.conv_expand.tmp_2",
    "elementwise_mul_0",
    "blocks.2.0.conv_pwl.tmp_0",
    "blocks.2.0.conv_pwl.tmp_1",
    "blocks.2.1.conv_pw.tmp_0",
    "blocks.2.1.conv_pw.tmp_1",
    "relu6_8.tmp_0",
    "blocks.2.1.conv_dw.tmp_0",
    "blocks.2.1.conv_dw.tmp_1",
    "relu6_9.tmp_0",
    "pool2d_1.tmp_0",
    "blocks.2.1.se.conv_reduce.tmp_0",
    "blocks.2.1.se.conv_reduce.tmp_1",
    "blocks.2.1.se.conv_reduce.tmp_2",
    "blocks.2.1.se.conv_expand.tmp_0",
    "blocks.2.1.se.conv_expand.tmp_1",
    "blocks.2.1.se.conv_expand.tmp_2",
    "elementwise_mul_1",
    "blocks.2.1.conv_pwl.tmp_0",
    "blocks.2.1.conv_pwl.tmp_1",
    "elementwise_add_1",
    "blocks.2.2.conv_pw.tmp_0",
    "blocks.2.2.conv_pw.tmp_1",
    "relu6_10.tmp_0",
    "blocks.2.2.conv_dw.tmp_0",
    "blocks.2.2.conv_dw.tmp_1",
    "relu6_11.tmp_0",
    "pool2d_2.tmp_0",
    "blocks.2.2.se.conv_reduce.tmp_0",
    "blocks.2.2.se.conv_reduce.tmp_1",
    "blocks.2.2.se.conv_reduce.tmp_2",
    "blocks.2.2.se.conv_expand.tmp_0",
    "blocks.2.2.se.conv_expand.tmp_1",
    "blocks.2.2.se.conv_expand.tmp_2",
    "elementwise_mul_2",
    "blocks.2.2.conv_pwl.tmp_0",
    "blocks.2.2.conv_pwl.tmp_1",
    "elementwise_add_2",
    "blocks.3.0.conv_pw.tmp_0",
    "blocks.3.0.conv_pw.tmp_1",
    "relu6_12.tmp_0",
    "blocks.3.0.conv_dw.tmp_0",
    "blocks.3.0.conv_dw.tmp_1",
    "relu6_13.tmp_0",
    "blocks.3.0.conv_pwl.tmp_0",
    "blocks.3.0.conv_pwl.tmp_1",
    "blocks.3.1.conv_pw.tmp_0",
    "blocks.3.1.conv_pw.tmp_1",
    "relu6_14.tmp_0",
    "blocks.3.1.conv_dw.tmp_0",
    "blocks.3.1.conv_dw.tmp_1",
    "relu6_15.tmp_0",
    "blocks.3.1.conv_pwl.tmp_0",
    "blocks.3.1.conv_pwl.tmp_1",
    "elementwise_add_3",
    "blocks.3.2.conv_pw.tmp_0",
    "blocks.3.2.conv_pw.tmp_1",
    "relu6_16.tmp_0",
    "blocks.3.2.conv_dw.tmp_0",
    "blocks.3.2.conv_dw.tmp_1",
    "relu6_17.tmp_0",
    "blocks.3.2.conv_pwl.tmp_0",
    "blocks.3.2.conv_pwl.tmp_1",
    "elementwise_add_4",
    "blocks.3.3.conv_pw.tmp_0",
    "blocks.3.3.conv_pw.tmp_1",
    "relu6_18.tmp_0",
    "blocks.3.3.conv_dw.tmp_0",
    "blocks.3.3.conv_dw.tmp_1",
    "relu6_19.tmp_0",
    "blocks.3.3.conv_pwl.tmp_0",
    "blocks.3.3.conv_pwl.tmp_1",
    "elementwise_add_5",
    "blocks.4.0.conv_pw.tmp_0",
    "blocks.4.0.conv_pw.tmp_1",
    "relu6_20.tmp_0",
    "blocks.4.0.conv_dw.tmp_0",
    "blocks.4.0.conv_dw.tmp_1",
    "relu6_21.tmp_0",
    "pool2d_3.tmp_0",
    "blocks.4.0.se.conv_reduce.tmp_0",
    "blocks.4.0.se.conv_reduce.tmp_1",
    "blocks.4.0.se.conv_reduce.tmp_2",
    "blocks.4.0.se.conv_expand.tmp_0",
    "blocks.4.0.se.conv_expand.tmp_1",
    "blocks.4.0.se.conv_expand.tmp_2",
    "elementwise_mul_3",
    "blocks.4.0.conv_pwl.tmp_0",
    "blocks.4.0.conv_pwl.tmp_1",
    "blocks.4.1.conv_pw.tmp_0",
    "blocks.4.1.conv_pw.tmp_1",
    "relu6_22.tmp_0",
    "blocks.4.1.conv_dw.tmp_0",
    "blocks.4.1.conv_dw.tmp_1",
    "relu6_23.tmp_0",
    "pool2d_4.tmp_0",
    "blocks.4.1.se.conv_reduce.tmp_0",
    "blocks.4.1.se.conv_reduce.tmp_1",
    "blocks.4.1.se.conv_reduce.tmp_2",
    "blocks.4.1.se.conv_expand.tmp_0",
    "blocks.4.1.se.conv_expand.tmp_1",
    "blocks.4.1.se.conv_expand.tmp_2",
    "elementwise_mul_4",
    "blocks.4.1.conv_pwl.tmp_0",
    "blocks.4.1.conv_pwl.tmp_1",
    "elementwise_add_6",
    "blocks.5.0.conv_pw.tmp_0",
    "blocks.5.0.conv_pw.tmp_1",
    "relu6_24.tmp_0",
    "blocks.5.0.conv_dw.tmp_0",
    "blocks.5.0.conv_dw.tmp_1",
    "relu6_25.tmp_0",
    "pool2d_5.tmp_0",
    "blocks.5.0.se.conv_reduce.tmp_0",
    "blocks.5.0.se.conv_reduce.tmp_1",
    "blocks.5.0.se.conv_reduce.tmp_2",
    "blocks.5.0.se.conv_expand.tmp_0",
    "blocks.5.0.se.conv_expand.tmp_1",
    "blocks.5.0.se.conv_expand.tmp_2",
    "elementwise_mul_5",
    "blocks.5.0.conv_pwl.tmp_0",
    "blocks.5.0.conv_pwl.tmp_1",
    "blocks.5.1.conv_pw.tmp_0",
    "blocks.5.1.conv_pw.tmp_1",
    "relu6_26.tmp_0",
    "blocks.5.1.conv_dw.tmp_0",
    "blocks.5.1.conv_dw.tmp_1",
    "relu6_27.tmp_0",
    "pool2d_6.tmp_0",
    "blocks.5.1.se.conv_reduce.tmp_0",
    "blocks.5.1.se.conv_reduce.tmp_1",
    "blocks.5.1.se.conv_reduce.tmp_2",
    "blocks.5.1.se.conv_expand.tmp_0",
    "blocks.5.1.se.conv_expand.tmp_1",
    "blocks.5.1.se.conv_expand.tmp_2",
    "elementwise_mul_6",
    "blocks.5.1.conv_pwl.tmp_0",
    "blocks.5.1.conv_pwl.tmp_1",
    "elementwise_add_7",
    "blocks.5.2.conv_pw.tmp_0",
    "blocks.5.2.conv_pw.tmp_1",
    "relu6_28.tmp_0",
    "blocks.5.2.conv_dw.tmp_0",
    "blocks.5.2.conv_dw.tmp_1",
    "relu6_29.tmp_0",
    "pool2d_7.tmp_0",
    "blocks.5.2.se.conv_reduce.tmp_0",
    "blocks.5.2.se.conv_reduce.tmp_1",
    "blocks.5.2.se.conv_reduce.tmp_2",
    "blocks.5.2.se.conv_expand.tmp_0",
    "blocks.5.2.se.conv_expand.tmp_1",
    "blocks.5.2.se.conv_expand.tmp_2",
    "elementwise_mul_7",
    "blocks.5.2.conv_pwl.tmp_0",
    "blocks.5.2.conv_pwl.tmp_1",
    "elementwise_add_8",
    "blocks.6.0.conv_pw.tmp_0",
    "blocks.6.0.conv_pw.tmp_1",
    "relu6_30.tmp_0",
    "blocks.6.0.conv_dw.tmp_0",
    "blocks.6.0.conv_dw.tmp_1",
    "relu6_31.tmp_0",
    "blocks.6.0.conv_pwl.tmp_0",
    "blocks.6.0.conv_pwl.tmp_1",
    "conv_head.tmp_0",
    "conv_head.tmp_1",
    "relu6_32.tmp_0",
    "pool2d_8.tmp_0",
    "classifier_cyf.tmp_0",
    "classifier_cyf.tmp_1",
    "save_infer_model/scale_0"};
// nanoyolo
std::vector<std::string> tensor_names_nanoyolo = {"image",
                                                  // "conv2d_0.tmp_0",
                                                  // "conv2d_0.tmp_1",
                                                  // "relu_0.tmp_0",
                                                  // "conv2d_1.tmp_0",
                                                  // "conv2d_1.tmp_1",
                                                  // "relu_1.tmp_0",
                                                  // "conv2d_2.tmp_0",
                                                  // "conv2d_2.tmp_1",
                                                  // "relu6_0.tmp_0",
                                                  // "conv2d_3.tmp_0",
                                                  // "conv2d_3.tmp_1",
                                                  // "relu6_1.tmp_0",
                                                  // "conv2d_4.tmp_0",
                                                  // "conv2d_4.tmp_1",
                                                  // "relu6_2.tmp_0",
                                                  // "conv2d_5.tmp_0",
                                                  // "conv2d_5.tmp_1",
                                                  // "tmp_0",
                                                  // "conv2d_6.tmp_0",
                                                  // "conv2d_6.tmp_1",
                                                  // "relu6_3.tmp_0",
                                                  // "conv2d_7.tmp_0",
                                                  // "conv2d_7.tmp_1",
                                                  // "relu6_4.tmp_0",
                                                  // "conv2d_8.tmp_0",
                                                  // "conv2d_8.tmp_1",
                                                  // "conv2d_9.tmp_0",
                                                  // "conv2d_9.tmp_1",
                                                  // "relu6_5.tmp_0",
                                                  // "conv2d_10.tmp_0",
                                                  // "conv2d_10.tmp_1",
                                                  // "relu6_6.tmp_0",
                                                  // "conv2d_11.tmp_0",
                                                  // "conv2d_11.tmp_1",
                                                  // "relu6_7.tmp_0",
                                                  // "conv2d_12.tmp_0",
                                                  // "conv2d_12.tmp_1",
                                                  // "tmp_1",
                                                  // "conv2d_13.tmp_0",
                                                  // "conv2d_13.tmp_1",
                                                  // "relu6_8.tmp_0",
                                                  // "conv2d_14.tmp_0",
                                                  // "conv2d_14.tmp_1",
                                                  // "relu6_9.tmp_0",
                                                  // "conv2d_15.tmp_0",
                                                  // "conv2d_15.tmp_1",
                                                  // "relu6_10.tmp_0",
                                                  // "conv2d_16.tmp_0",
                                                  // "conv2d_16.tmp_1",
                                                  // "tmp_2",
                                                  // "conv2d_17.tmp_0",
                                                  // "conv2d_17.tmp_1",
                                                  // "relu6_11.tmp_0",
                                                  // "conv2d_18.tmp_0",
                                                  // "conv2d_18.tmp_1",
                                                  // "relu6_12.tmp_0",
                                                  // "conv2d_19.tmp_0",
                                                  // "conv2d_19.tmp_1",
                                                  // "conv2d_20.tmp_0",
                                                  // "conv2d_20.tmp_1",
                                                  // "relu6_13.tmp_0",
                                                  // "conv2d_21.tmp_0",
                                                  // "conv2d_21.tmp_1",
                                                  // "relu6_14.tmp_0",
                                                  // "conv2d_22.tmp_0",
                                                  // "conv2d_22.tmp_1",
                                                  // "relu6_15.tmp_0",
                                                  // "conv2d_23.tmp_0",
                                                  // "conv2d_23.tmp_1",
                                                  // "tmp_3",
                                                  // "conv2d_24.tmp_0",
                                                  // "conv2d_24.tmp_1",
                                                  // "relu_2.tmp_0",
                                                  // "pool2d_0.tmp_0",
                                                  // "conv2d_25.tmp_0",
                                                  // "relu_3.tmp_0",
                                                  // "conv2d_26.tmp_0",
                                                  // "sigmoid_0.tmp_0",
                                                  // "elementwise_mul_0",
                                                  // "conv2d_27.tmp_0",
                                                  // "conv2d_27.tmp_1",
                                                  // "relu6_16.tmp_0",
                                                  // "conv2d_28.tmp_0",
                                                  // "conv2d_28.tmp_1",
                                                  // "relu6_17.tmp_0",
                                                  // "conv2d_29.tmp_0",
                                                  // "conv2d_29.tmp_1",
                                                  // "relu6_18.tmp_0",
                                                  // "conv2d_30.tmp_0",
                                                  // "conv2d_30.tmp_1",
                                                  // "tmp_4",
                                                  // "conv2d_31.tmp_0",
                                                  // "conv2d_31.tmp_1",
                                                  // "relu6_19.tmp_0",
                                                  // "conv2d_32.tmp_0",
                                                  // "conv2d_32.tmp_1",
                                                  // "relu6_20.tmp_0",
                                                  // "conv2d_33.tmp_0",
                                                  // "conv2d_33.tmp_1",
                                                  // "relu6_21.tmp_0",
                                                  // "conv2d_34.tmp_0",
                                                  // "conv2d_34.tmp_1",
                                                  // "tmp_5",
                                                  // "conv2d_35.tmp_0",
                                                  // "conv2d_35.tmp_1",
                                                  // "relu6_22.tmp_0",
                                                  // "conv2d_36.tmp_0",
                                                  // "conv2d_36.tmp_1",
                                                  // "relu6_23.tmp_0",
                                                  // "conv2d_37.tmp_0",
                                                  // "conv2d_37.tmp_1",
                                                  // "relu6_24.tmp_0",
                                                  // "conv2d_38.tmp_0",
                                                  // "conv2d_38.tmp_1",
                                                  // "tmp_6",
                                                  // "conv2d_39.tmp_0",
                                                  // "conv2d_39.tmp_1",
                                                  // "relu6_25.tmp_0",
                                                  // "conv2d_40.tmp_0",
                                                  // "conv2d_40.tmp_1",
                                                  // "relu6_26.tmp_0",
                                                  // "conv2d_41.tmp_0",
                                                  // "conv2d_41.tmp_1",
                                                  // "conv2d_42.tmp_0",
                                                  // "conv2d_42.tmp_1",
                                                  // "relu6_27.tmp_0",
                                                  // "conv2d_43.tmp_0",
                                                  // "conv2d_43.tmp_1",
                                                  // "relu6_28.tmp_0",
                                                  // "conv2d_44.tmp_0",
                                                  // "conv2d_44.tmp_1",
                                                  // "relu6_29.tmp_0",
                                                  // "conv2d_45.tmp_0",
                                                  // "conv2d_45.tmp_1",
                                                  // "tmp_7",
                                                  // "conv2d_46.tmp_0",
                                                  // "conv2d_46.tmp_1",
                                                  // "relu6_30.tmp_0",
                                                  // "conv2d_47.tmp_0",
                                                  // "conv2d_47.tmp_1",
                                                  // "relu6_31.tmp_0",
                                                  // "conv2d_48.tmp_0",
                                                  // "conv2d_48.tmp_1",
                                                  // "relu6_32.tmp_0",
                                                  // "conv2d_49.tmp_0",
                                                  // "conv2d_49.tmp_1",
                                                  // "tmp_8",
                                                  // "conv2d_50.tmp_0",
                                                  // "conv2d_50.tmp_1",
                                                  // "relu6_33.tmp_0",
                                                  // "conv2d_51.tmp_0",
                                                  // "conv2d_51.tmp_1",
                                                  // "relu6_34.tmp_0",
                                                  // "conv2d_52.tmp_0",
                                                  // "conv2d_52.tmp_1",
                                                  // "relu6_35.tmp_0",
                                                  // "conv2d_53.tmp_0",
                                                  // "conv2d_53.tmp_1",
                                                  // "tmp_9",
                                                  // "conv2d_54.tmp_0",
                                                  // "conv2d_54.tmp_1",
                                                  // "relu6_36.tmp_0",
                                                  // "conv2d_55.tmp_0",
                                                  // "conv2d_55.tmp_1",
                                                  // "relu6_37.tmp_0",
                                                  // "conv2d_56.tmp_0",
                                                  // "conv2d_56.tmp_1",
                                                  // "relu6_38.tmp_0",
                                                  // "conv2d_57.tmp_0",
                                                  // "conv2d_57.tmp_1",
                                                  // "tmp_10",
                                                  // "conv2d_58.tmp_0",
                                                  // "conv2d_58.tmp_1",
                                                  // "relu6_39.tmp_0",
                                                  // "conv2d_59.tmp_0",
                                                  // "conv2d_59.tmp_1",
                                                  // "relu6_40.tmp_0",
                                                  // "conv2d_60.tmp_0",
                                                  // "conv2d_60.tmp_1",
                                                  // "relu6_41.tmp_0",
                                                  // "conv2d_61.tmp_0",
                                                  // "conv2d_61.tmp_1",
                                                  // "tmp_11",
                                                  // "conv2d_62.tmp_0",
                                                  // "conv2d_62.tmp_1",
                                                  // "relu6_42.tmp_0",
                                                  // "conv2d_63.tmp_0",
                                                  // "conv2d_63.tmp_1",
                                                  // "relu6_43.tmp_0",
                                                  // "conv2d_64.tmp_0",
                                                  // "conv2d_64.tmp_1",
                                                  // "relu6_44.tmp_0",
                                                  // "conv2d_65.tmp_0",
                                                  // "conv2d_65.tmp_1",
                                                  // "tmp_12",
                                                  // "conv2d_66.tmp_0",
                                                  // "conv2d_66.tmp_1",
                                                  // "relu6_45.tmp_0",
                                                  // "conv2d_67.tmp_0",
                                                  // "conv2d_67.tmp_1",
                                                  // "relu6_46.tmp_0",
                                                  // "conv2d_68.tmp_0",
                                                  // "conv2d_68.tmp_1",
                                                  // "relu6_47.tmp_0",
                                                  // "conv2d_69.tmp_0",
                                                  // "conv2d_69.tmp_1",
                                                  // "tmp_13",
                                                  // "conv2d_70.tmp_0",
                                                  // "conv2d_70.tmp_1",
                                                  // "relu6_48.tmp_0",
                                                  // "conv2d_71.tmp_0",
                                                  // "conv2d_71.tmp_1",
                                                  // "relu6_49.tmp_0",
                                                  // "conv2d_72.tmp_0",
                                                  // "conv2d_72.tmp_1",
                                                  // "relu6_50.tmp_0",
                                                  // "conv2d_73.tmp_0",
                                                  // "conv2d_73.tmp_1",
                                                  // "tmp_14",
                                                  // "conv2d_74.tmp_0",
                                                  // "conv2d_74.tmp_1",
                                                  // "relu6_51.tmp_0",
                                                  // "conv2d_75.tmp_0",
                                                  // "conv2d_75.tmp_1",
                                                  // "relu6_52.tmp_0",
                                                  // "conv2d_76.tmp_0",
                                                  // "conv2d_76.tmp_1",
                                                  // "conv2d_77.tmp_0",
                                                  // "conv2d_77.tmp_1",
                                                  // "relu6_53.tmp_0",
                                                  // "conv2d_78.tmp_0",
                                                  // "conv2d_78.tmp_1",
                                                  // "relu6_54.tmp_0",
                                                  // "conv2d_79.tmp_0",
                                                  // "conv2d_79.tmp_1",
                                                  // "relu6_55.tmp_0",
                                                  // "conv2d_80.tmp_0",
                                                  // "conv2d_80.tmp_1",
                                                  // "tmp_15",
                                                  // "conv2d_81.tmp_0",
                                                  // "conv2d_81.tmp_1",
                                                  // "relu_4.tmp_0",
                                                  // "conv2d_82.tmp_0",
                                                  // "conv2d_82.tmp_1",
                                                  // "relu6_56.tmp_0",
                                                  // "conv2d_83.tmp_0",
                                                  // "conv2d_83.tmp_1",
                                                  // "relu6_57.tmp_0",
                                                  // "conv2d_84.tmp_0",
                                                  // "conv2d_84.tmp_1",
                                                  // "conv2d_85.tmp_0",
                                                  // "conv2d_85.tmp_1",
                                                  // "relu6_58.tmp_0",
                                                  // "conv2d_86.tmp_0",
                                                  // "conv2d_86.tmp_1",
                                                  // "relu6_59.tmp_0",
                                                  // "conv2d_87.tmp_0",
                                                  // "conv2d_87.tmp_1",
                                                  // "relu6_60.tmp_0",
                                                  // "conv2d_88.tmp_0",
                                                  // "conv2d_88.tmp_1",
                                                  // "conv2d_89.tmp_0",
                                                  // "conv2d_89.tmp_1",
                                                  // "relu_5.tmp_0",
                                                  // "conv2d_90.tmp_0",
                                                  // "conv2d_90.tmp_1",
                                                  // "relu6_61.tmp_0",
                                                  // "conv2d_91.tmp_0",
                                                  // "conv2d_91.tmp_1",
                                                  // "relu6_62.tmp_0",
                                                  // "conv2d_92.tmp_0",
                                                  // "conv2d_92.tmp_1",
                                                  // "conv2d_93.tmp_0",
                                                  // "conv2d_93.tmp_1",
                                                  // "relu_6.tmp_0",
                                                  // "h2_02_upsample.tmp_0",
                                                  // "concat_0.tmp_0",
                                                  // "conv2d_94.tmp_0",
                                                  // "conv2d_94.tmp_1",
                                                  // "relu6_63.tmp_0",
                                                  // "conv2d_95.tmp_0",
                                                  // "conv2d_95.tmp_1",
                                                  // "relu6_64.tmp_0",
                                                  // "conv2d_96.tmp_0",
                                                  // "conv2d_96.tmp_1",
                                                  // "relu6_65.tmp_0",
                                                  // "conv2d_97.tmp_0",
                                                  // "conv2d_97.tmp_1",
                                                  // "conv2d_98.tmp_0",
                                                  // "conv2d_98.tmp_1",
                                                  // "relu6_66.tmp_0",
                                                  // "conv2d_99.tmp_0",
                                                  // "conv2d_99.tmp_1",
                                                  // "relu6_67.tmp_0",
                                                  // "conv2d_100.tmp_0",
                                                  // "conv2d_100.tmp_1",
                                                  // "relu6_68.tmp_0",
                                                  // "conv2d_101.tmp_0",
                                                  // "conv2d_101.tmp_1",
                                                  // "conv2d_102.tmp_0",
                                                  // "conv2d_102.tmp_1",
                                                  // "relu_7.tmp_0",
                                                  // "conv2d_103.tmp_0",
                                                  // "conv2d_103.tmp_1",
                                                  // "relu6_69.tmp_0",
                                                  // "conv2d_104.tmp_0",
                                                  // "conv2d_104.tmp_1",
                                                  // "relu6_70.tmp_0",
                                                  // "conv2d_105.tmp_0",
                                                  // "conv2d_105.tmp_1",
                                                  // "conv2d_106.tmp_0",
                                                  // "conv2d_106.tmp_1",
                                                  // "relu_8.tmp_0",
                                                  // "h1_02_upsample.tmp_0",
                                                  // "concat_1.tmp_0",
                                                  // "conv2d_107.tmp_0",
                                                  // "conv2d_107.tmp_1",
                                                  // "relu6_71.tmp_0",
                                                  // "conv2d_108.tmp_0",
                                                  // "conv2d_108.tmp_1",
                                                  // "relu6_72.tmp_0",
                                                  // "conv2d_109.tmp_0",
                                                  // "conv2d_109.tmp_1",
                                                  // "relu6_73.tmp_0",
                                                  // "conv2d_110.tmp_0",
                                                  // "conv2d_110.tmp_1",
                                                  // "conv2d_111.tmp_0",
                                                  // "conv2d_111.tmp_1",
                                                  // "relu6_74.tmp_0",
                                                  // "conv2d_112.tmp_0",
                                                  // "conv2d_112.tmp_1",
                                                  // "relu6_75.tmp_0",
                                                  // "conv2d_113.tmp_0",
                                                  // "conv2d_113.tmp_1",
                                                  // "relu6_76.tmp_0",
                                                  // "conv2d_114.tmp_0",
                                                  // "conv2d_114.tmp_1",
                                                  // "conv2d_115.tmp_0",
                                                  // "conv2d_115.tmp_1",
                                                  // "relu6_77.tmp_0",
                                                  // "conv2d_116.tmp_0",
                                                  // "conv2d_116.tmp_1",
                                                  // "relu6_78.tmp_0",
                                                  // "conv2d_117.tmp_0",
                                                  // "conv2d_117.tmp_1",
                                                  // "relu6_79.tmp_0",
                                                  // "conv2d_118.tmp_0",
                                                  // "conv2d_118.tmp_1",
                                                  // "conv2d_119.tmp_0",
                                                  // "conv2d_119.tmp_1",
                                                  // "conv2d_120.tmp_0",
                                                  // "conv2d_120.tmp_1",
                                                  // "conv2d_121.tmp_0",
                                                  // "conv2d_121.tmp_1",
                                                  // "conv2d_122.tmp_0",
                                                  // "conv2d_122.tmp_1",
                                                  // "conv2d_122.tmp_2",
                                                  // "conv2d_123.tmp_0",
                                                  // "conv2d_123.tmp_1",
                                                  // "conv2d_123.tmp_2",
                                                  // "conv2d_124.tmp_0",
                                                  // "conv2d_124.tmp_1",
                                                  // "conv2d_124.tmp_2",
                                                  // "reshape2_0.tmp_0",
                                                  // "reshape2_1.tmp_0",
                                                  // "reshape2_2.tmp_0",
                                                  // "reshape2_3.tmp_0",
                                                  // "reshape2_4.tmp_0",
                                                  // "reshape2_5.tmp_0",
                                                  // "concat_2.tmp_0",
                                                  // "concat_3.tmp_0",
                                                  // "concat_4.tmp_0",
                                                  "save_infer_model/scale_0"};

std::vector<std::string> tensor_names = tensor_names_nanoyolo;
void fill_tensor_from_path(const std::string& file, int size, float* addr) {
  std::ifstream in(file, std::ios::in);
  for (int i = 0; i < size; i++) {
    float num;
    in >> num;
    // LOG(INFO) << "inputs[ " << i << " ] " << num;
    addr[i] = num;
  }
  in.close();
}
void stride_print(int numel, float* result) {
  int stride = numel / 20;
  stride = stride > 0 ? stride : 1;
  for (int i = 0; i < numel; i += stride) {
    LOG(INFO) << "Tensor : " << result[i];
  }
}
void stride_print_const(int numel, const float* result) {
  int stride = numel / 20;
  stride = stride > 0 ? stride : 1;
  for (int i = 0; i < numel; i += stride) {
    LOG(INFO) << "Tensor : " << result[i];
  }
}

void SplitString(const std::string& s,
                 const std::string& c,
                 std::vector<std::string>& v) {
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while (std::string::npos != pos2) {
    v.push_back(s.substr(pos1, pos2 - pos1));

    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length()) v.push_back(s.substr(pos1));
}
void TestModelOpenCL(const std::vector<Place>& valid_places,
                     const std::string& model_dir = FLAGS_model_dir,
                     bool save_model = false) {
  LOG(INFO) << "FLAGS_check_nb:  " << FLAGS_check_nb;

  auto tensor_mean_cl = [](const Tensor* in,
                           PrecisionType ptype,
                           std::string name = "inst") -> double {
    if (!in->data<int8_t>()) {
      return -99999;
    }
    double sum = 0.;
    // profile opencl
    switch (ptype) {
      case PRECISION(kFloat): {
        paddle::lite::CLImageConverterDefault default_convertor;
        DDim out_image_shape =
            default_convertor.InitImageDimInfoWith(in->dims());
        int out_image_width = out_image_shape[0];
        int out_image_height = out_image_shape[1];

        const size_t cl_image2d_row_pitch{0};
        const size_t cl_image2d_slice_pitch{0};
        VLOG(5) << "out_image_shape: " << out_image_shape[0] << "  "
                << out_image_shape[1];
        std::vector<uint16_t> out_image_v(out_image_shape.production() *
                                          4);  // 4 :RGBA
        std::vector<float> output_v(in->dims().production());
        auto* indata = in->data<float, cl::Image2D>();
        VLOG(5) << "indata addr: " << indata;
        if (indata == nullptr) {
          return -1;
        }
        TargetWrapperCL::ImgcpySync(out_image_v.data(),
                                    in->data<uint16_t, cl::Image2D>(),
                                    out_image_width,
                                    out_image_height,
                                    cl_image2d_row_pitch,
                                    cl_image2d_slice_pitch,
                                    IoDirection::DtoH);
        // LOG(INFO) << "out_image_v: ";
        // stride_print(out_image_v.size(), out_image_v.data());
        default_convertor.ImageToNCHW(
            out_image_v.data(), output_v.data(), out_image_shape, in->dims());
        // LOG(INFO) << "output_v: ";
        // stride_print(output_v.size(), output_v.data());
        for (size_t i = 0; i < output_v.size(); i++) {
          sum += output_v[i];
        }

        return sum / in->numel();
      }

      default:
        LOG(INFO) << "opencl unsupport data type: " << PrecisionToStr(ptype);
        return 0.;
    }
  };

  auto tensor_string = [](
      const Tensor* in, PrecisionType ptype, std::string name = "inst") -> int {
    if (!in->data<int8_t>()) {
      std::cout << "lite-auto-test"
                << " var " << name << " NOTGET " << std::endl;
      return -1;
    }
    double sum = 0.;
    // profile opencl
    switch (ptype) {
      case PRECISION(kFloat): {
        paddle::lite::CLImageConverterDefault default_convertor;
        DDim out_image_shape =
            default_convertor.InitImageDimInfoWith(in->dims());
        int out_image_width = out_image_shape[0];
        int out_image_height = out_image_shape[1];

        const size_t cl_image2d_row_pitch{0};
        const size_t cl_image2d_slice_pitch{0};
        VLOG(5) << "out_image_shape: " << out_image_shape[0] << "  "
                << out_image_shape[1];
        std::vector<uint16_t> out_image_v(out_image_shape.production() *
                                          4);  // 4 :RGBA
        auto in_dims = in->dims();
        int len = in_dims.production();
        std::vector<float> output_v(len);
        auto* indata = in->data<float, cl::Image2D>();
        VLOG(5) << "indata addr: " << indata;
        if (indata == nullptr) {
          std::cout << "lite-auto-test"
                    << " var " << name << " NOTGET " << std::endl;
          return -2;
        }
        TargetWrapperCL::ImgcpySync(out_image_v.data(),
                                    in->data<uint16_t, cl::Image2D>(),
                                    out_image_width,
                                    out_image_height,
                                    cl_image2d_row_pitch,
                                    cl_image2d_slice_pitch,
                                    IoDirection::DtoH);
        // LOG(INFO) << "out_image_v: ";
        // stride_print(out_image_v.size(), out_image_v.data());
        default_convertor.ImageToNCHW(
            out_image_v.data(), output_v.data(), out_image_shape, in_dims);
        // LOG(INFO) << "output_v: ";
        // stride_print(output_v.size(), output_v.data());
        for (size_t i = 0; i < output_v.size(); i++) {
          sum += output_v[i];
        }
        std::string sample = "";
        if (FLAGS_check_shape) {
          for (int i = 0; i < len; i++) {
            sample += " " + std::to_string(in_dims[i]);
          }
        }
        int sample_step = 1;
        if (!FLAGS_is_sample_step) {
          sample_step = len / FLAGS_sample_num;
        } else {
          sample_step = FLAGS_sample_step;
        }
        if (sample_step <= 0) {
          sample_step = 1;
        }
        double sum = 0;
        for (int i = 0; i < len; i += sample_step) {
          float datai = output_v[i];
          sum += datai;
          sample += " " + std::to_string(output_v[i]);
        }
        std::cout << "lite-auto-test"
                  << " var " << name << sample << std::endl;
        return -3;
      }

      default:
        LOG(INFO) << "opencl unsupport data type: " << PrecisionToStr(ptype);
        std::cout << "lite-auto-test"
                  << " var " << name << " NOTGET " << std::endl;
        return -4;
    }
  };
  const std::string feeds_dir("/data/local/tmp/opencl/");
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);
  lite::Predictor predictor;
  if (FLAGS_check_nb) {
    // predictor.Build(
    //     model_dir + "lens_nanoyolo_opencl.nb", "", "", valid_places);
    predictor.Build(model_dir + "lens_nanoyolo_opencl.nb",
                    "",
                    "",
                    valid_places,
                    {},
                    lite_api::LiteModelType::kNaiveBuffer);
  } else {
    predictor.Build(model_dir, "", "", valid_places);
  }

  const std::string input_flag = std::string(FLAGS_input_file.c_str());
  // 多输入:
  if (input_flag.find(":") != input_flag.npos) {
    const std::string split_key1 = ";";
    const std::string split_key2 = ":";
    const std::string split_key3 = "_";
    std::vector<std::string> feeds;
    SplitString(input_flag, split_key1, feeds);

    for (size_t i = 0; i < feeds.size(); i++) {
      std::vector<std::string> name_dims;
      SplitString(feeds[i], split_key2, name_dims);
      std::string name = name_dims[0];
      std::string dim_str = name_dims[1];
      std::vector<std::string> dims_s_v;
      // LOG(INFO) << "dims_s_v:  " << dims_s_v;
      SplitString(dim_str, split_key3, dims_s_v);
      int N = std::stoi(dims_s_v[0]);
      int C = std::stoi(dims_s_v[1]);
      int H = std::stoi(dims_s_v[2]);
      int W = std::stoi(dims_s_v[3]);
      // init tensor
      auto* input_tensor = predictor.GetInputByName(name);
      input_tensor->Resize(DDim(std::vector<DDim::value_type>({N, C, H, W})));
      LOG(INFO) << "feed name: " << name << "  dims:" << N << C << H << W;
      auto* data = input_tensor->mutable_data<float>();
      auto input_size = input_tensor->dims().production();
      fill_tensor_from_path(feeds_dir + name, input_size, data);
      LOG(INFO) << "inputs - " << name;
      stride_print(input_size, data);
    }

  } else {
    LOG(FATAL) << "did not get input input_flag : " << input_flag;
  }

  // std::unordered_map<std::string, std::unique_ptr<Variable>> vars =
  //     predictor.GetScope()->GetVars();
  // std::vector<std::string> var_names = predictor.GetScope()->LocalVarNames();

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor.Run();
  }

// 分析输出, 但是这种方式名字似乎不同,不可用
#if 0
  const std::string output_flag = std::string(FLAGS_output_file.c_str());
  LOG(INFO) << "output_flag:  " << output_flag;
  // 多输出:
  if (output_flag.find(":") != output_flag.npos) {
    const std::string split_key1 = ";";
    const std::string split_key2 = ":";
    const std::string split_key3 = "_";
    std::vector<std::string> fetchs;
    SplitString(output_flag, split_key1, fetchs);
    LOG(INFO) << "fetchs.size():  " << fetchs.size();
    for (size_t i = 0; i < fetchs.size(); i++) {
      std::vector<std::string> name_dims;
      SplitString(fetchs[i], split_key2, name_dims);
      std::string name = name_dims[0];
      std::string dim_str = name_dims[1];
      std::vector<std::string> dims_s_v;
      SplitString(dim_str, split_key3, dims_s_v);
      LOG(INFO) << "dims_s_v.size():  " << dims_s_v.size();

      // init tensor
      // predictor.GetOutput(0);
      auto* output_tensor =
          predictor.exec_scope_->FindVar(name)->GetMutable<lite::Tensor>();
      LOG(INFO) << "FIND OUTPUT TENSOR SUCCESS ";

      // output_tensor->Resize(DDim(std::vector<DDim::value_type>({N, C, H,
      // W})));

      LOG(INFO) << "fetch name: " << name;

      auto* data = output_tensor->data<float>();
      // auto output_size = output_tensor->dims().production();
      // fill_tensor_from_path(feeds_dir + name, output_size, data);
      // LOG(INFO) << "outputs - " << name;
      // stride_print(output_size, data);
      auto output_dims = output_tensor->dims();
      int len = output_dims.production();
      std::string sample = "";
      if (FLAGS_check_shape) {
        for (int i = 0; i < len; i++) {
          sample += " " + std::to_string(output_dims[i]);
        }
      }
      int sample_step = 1;
      if (!FLAGS_is_sample_step) {
        sample_step = len / FLAGS_sample_num;
      } else {
        sample_step = FLAGS_sample_step;
      }
      if (sample_step <= 0) {
        sample_step = 1;
      }
      double sum = 0;
      for (int i = 0; i < len; i += sample_step) {
        float datai = data[i];
        sum += datai;
        sample += " " + std::to_string(data[i]);
      }
      std::cout << "lite-auto-test"
                << " var " << name << sample << std::endl;
    }
  }
#endif

  auto fetch_names = predictor.GetOutputNames();
  auto fetch_name0 = fetch_names[0];
  for (size_t i = 0; i < fetch_names.size(); i++) {
    LOG(INFO) << "fetch name : " << i << "  " << fetch_names[i];
  }
  for (auto&& name : tensor_names) {
    VLOG(5) << "name" << name;

    auto* tensor = predictor.GetTensor(name);
    VLOG(5) << "tensor addr: " << tensor;
    // cl mem && 不是排除的
    if (tensor->is_cl_memory &&
        std::find(exclude_names.begin(), exclude_names.end(), name) ==
            exclude_names.end()) {
      VLOG(5) << "is opencl mem" << tensor->is_cl_memory;
      double mean = tensor_mean_cl(tensor, PrecisionType::kFloat);
      VLOG(5) << "tensor->numel()" << tensor->numel();
      // printf("%25.s   %5.5f \n", name.c_str(), mean);
      std::cout << std::setw(30) << name << std::setw(30) << mean
                << std::setw(30) << tensor->dims() << std::endl;
    } else {
      std::cout << std::setw(30) << "not get" << std::setw(30) << "NAN"
                << std::setw(30) << tensor->dims() << std::endl;
    }
    // 结果对比脚本
    if (FLAGS_checkscript) {
      if (tensor->is_cl_memory &&
          std::find(exclude_names.begin(), exclude_names.end(), name) ==
              exclude_names.end()) {
        tensor_string(tensor, PrecisionType::kFloat, name);
      } else if (fetch_name0.find(name) != std::string::npos) {
        auto output = predictor.GetOutput(0);
        auto* data = output->data<float>();
        // auto output_size = output_tensor->dims().production();
        // fill_tensor_from_path(feeds_dir + name, output_size, data);
        // LOG(INFO) << "outputs - " << name;
        // stride_print(output_size, data);
        auto output_dims = output->dims();
        int len = output_dims.production();
        std::string sample = "";
        if (FLAGS_check_shape) {
          for (int i = 0; i < len; i++) {
            sample += " " + std::to_string(output_dims[i]);
          }
        }
        int sample_step = 1;
        if (!FLAGS_is_sample_step) {
          sample_step = len / FLAGS_sample_num;
        } else {
          sample_step = FLAGS_sample_step;
        }
        if (sample_step <= 0) {
          sample_step = 1;
        }
        double sum = 0;
        for (int i = 0; i < len; i += sample_step) {
          float datai = data[i];
          sum += datai;
          sample += " " + std::to_string(data[i]);
        }
        std::cout << "lite-auto-test"
                  << " var " << name << sample << std::endl;
      } else {
        std::cout << "lite-auto-test"
                  << " var " << name << " NOTGET " << std::endl;
      }
    }
  }

  if (!FLAGS_check_nb) {
    predictor.SaveModel("/data/local/tmp/opencloptmodel",
                        lite_api::LiteModelType::kNaiveBuffer,
                        false);
  }

  LOG(INFO) << "input shape(NCHW):" << FLAGS_N << " " << FLAGS_C << " "
            << FLAGS_H << " " << FLAGS_W;
  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";
}

TEST(TEST_NET_CP, test_opencl) {
  std::vector<Place> valid_places({
      Place{TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault)},
      Place{TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kImageDefault)},
      Place{TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)},
      TARGET(kARM),  // enable kARM CPU kernel when no opencl kernel
  });

  TestModelOpenCL(valid_places);
}

void TestModelARM(const std::vector<Place>& valid_places,
                  const std::string& model_dir = FLAGS_model_dir,
                  bool save_model = false) {
  const std::string feeds_dir("/data/local/tmp/opencl/");
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(lite_api::LITE_POWER_NO_BIND, FLAGS_threads);
  lite::Predictor predictor;

  predictor.Build(model_dir, "", "", valid_places);
  const std::string input_flag = std::string(FLAGS_input_file.c_str());
  // 多输入:
  if (input_flag.find(":") != input_flag.npos) {
    const std::string split_key1 = ";";
    const std::string split_key2 = ":";
    const std::string split_key3 = "_";
    std::vector<std::string> feeds;
    SplitString(input_flag, split_key1, feeds);

    for (size_t i = 0; i < feeds.size(); i++) {
      std::vector<std::string> name_dims;
      SplitString(feeds[i], split_key2, name_dims);
      std::string name = name_dims[0];
      std::string dim_str = name_dims[1];
      std::vector<std::string> dims_s_v;
      // LOG(INFO) << "dims_s_v:  " << dims_s_v;
      SplitString(dim_str, split_key3, dims_s_v);
      int N = std::stoi(dims_s_v[0]);
      int C = std::stoi(dims_s_v[1]);
      int H = std::stoi(dims_s_v[2]);
      int W = std::stoi(dims_s_v[3]);
      // init tensor
      auto* input_tensor = predictor.GetInputByName(name);
      input_tensor->Resize(DDim(std::vector<DDim::value_type>({N, C, H, W})));
      LOG(INFO) << "feed name: " << name << "  dims:" << N << C << H << W;
      auto* data = input_tensor->mutable_data<float>();
      auto input_size = input_tensor->dims().production();
      fill_tensor_from_path(feeds_dir + name, input_size, data);
      LOG(INFO) << "inputs - " << name;
      stride_print(input_size, data);
    }
  } else {
    LOG(FATAL) << "did not get input ///";
  }

  // std::unordered_map<std::string, std::unique_ptr<Variable>> vars =
  //     predictor.GetScope()->GetVars();
  // std::vector<std::string> var_names = predictor.GetScope()->LocalVarNames();

  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor.Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    predictor.Run();
  }

  if (save_model) {
    LOG(INFO) << "Save optimized model to " << FLAGS_optimized_model;
    predictor.SaveModel(FLAGS_optimized_model);
  }
  predictor.SaveModel("/data/local/tmp/armoptmodel",
                      lite_api::LiteModelType::kNaiveBuffer,
                      false);
  LOG(INFO) << "input shape(NCHW):" << FLAGS_N << " " << FLAGS_C << " "
            << FLAGS_H << " " << FLAGS_W;
  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  auto* out = predictor.GetOutput(0);
  const auto* out_data = out->data<float>();
  std::vector<float> fluid_result(out->numel());

  LOG(INFO) << "lite 结果: ";
  stride_print_const(out->numel(), out_data);

  fill_tensor_from_path(FLAGS_output_file, out->numel(), fluid_result.data());
  LOG(INFO) << "fluid 结果: ";
  stride_print(out->numel(), fluid_result.data());
}
TEST(TEST_NET_CP, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kARM), PRECISION(kFloat)},
  });
  LOG(INFO) << "FLAGS_check_nb:  " << FLAGS_check_nb;
  if (FLAGS_check_nb) {
    return;
  }
  TestModelARM(valid_places);
}

}  // namespace lite
}  // namespace paddle
