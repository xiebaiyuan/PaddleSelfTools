#!/usr/bin/python3

import sys
import os
import time
import shutil
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddleslim.analysis import flops
paddle.enable_static()

def get_dir_size(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum(
            [os.path.getsize(os.path.join(root, name)) for name in files])
    return size


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ./{} <model_dir>".format(
            os.path.basename(__file__)))
        exit(0)

    # input parameters
    model_dir = sys.argv[1]  #whole dir
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    save_model_dir = "".join([model_dir, "/", "saved-", timestamp])
    exe = fluid.Executor(fluid.CPUPlace())

    print("[INFO] model_dir:{}".format(model_dir))

    # set params, model file names
    params_filename = ""
    model_filename = ""
    if os.path.exists(model_dir + "/model.pdmodel"):
        model_filename = "model.pdmodel"
        params_filename = "model.pdiparams"
    elif os.path.exists(model_dir + "/weights"):
        params_filename = "weights"
    elif os.path.exists(model_dir + "/params"):
        params_filename = "params"
    else:
        print("[ERROR] No model parameters file `weights` or `params` found")
        exit(0)

    print("[INFO] model_filename:{}".format(model_filename))
    print("[INFO] params_filename:{}".format(params_filename))

    # load model
    [inference_program, feed_target_names, fetch_targets] = \
      fluid.io.load_inference_model(dirname=model_dir, executor=exe, model_filename=model_filename, params_filename=params_filename)

    blocks = inference_program.blocks
    op_num = 0
    for block in blocks:
        op_num += len(block.ops)
        pass
    # print(inference_program.blocks)
    print("FLOPs: {:.6e} {:.3f}Mb  ops: {}".format(
        flops(inference_program),
        get_dir_size(model_dir) / 1024 / 1024, op_num))
