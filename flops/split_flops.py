#!/usr/bin/python3

import sys
import os
import time
import shutil
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddleslim.analysis import flops

DEBUG = False


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
    os.mkdir(save_model_dir)
    if os.path.exists(model_dir + "/model"):
        model_filename = "model"
    elif os.path.exists(model_dir + "/__model__"):
        model_filename = "__model__"
    else:
        print("[ERROR] No model file `model` or `__model__` found")
        exit(0)

    exe = fluid.Executor(fluid.CPUPlace())
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(dirname=model_dir,
                                                    executor=exe)

    blocks = inference_program.blocks
    op_num = 0
    for block in blocks:
        op_num += len(block.ops)
        pass
    # print(inference_program.blocks)
    print("FLOPs: {:.6e} {:.3f}Mb  ops: {}".format(
        flops(inference_program),
        get_dir_size(model_dir) / 1024 / 1024, op_num))