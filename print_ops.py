#!/usr/bin/python3

import sys
import os
import time
import shutil
import paddle.fluid as fluid

DEBUG = False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ./{} <model_dir>".format(
            os.path.basename(__file__)))
        exit(0)

    # input parameters
    model_dir = sys.argv[1]  #whole dir
    # timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # save_model_dir = "".join([model_dir, "/", "saved-", timestamp])
    # os.mkdir(save_model_dir)
    if os.path.exists(model_dir + "/model"):
        model_filename = "model"
    elif os.path.exists(model_dir + "/__model__"):
        model_filename = "__model__"
    else:
        print("[ERROR] No model file `model` or `__model__` found")
        exit(0)
    # exe = fluid.Executor(fluid.CPUPlace())
    # [inference_program, feed_target_names,
    #  fetch_targets] = fluid.io.load_inference_model(
    #      dirname=model_dir, executor=exe)

    for target_list in inference_program.blocks[0].ops:
        print(target_list.type)
