#!/usr/bin/python3

import sys
import os
import time
import shutil
import paddle.fluid as fluid
import paddle
DEBUG = False

paddle.enable_static()
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 ./{} <model_dir>".format(
            os.path.basename(__file__)))
        exit(0)

    # input parameters
    model_dir = sys.argv[1]
    if os.path.isdir(model_dir):
        dir_basename = os.path.basename(model_dir)
    model_name = sys.argv[2]
    param_name = sys.argv[3]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_model_dir = "".join(
        [model_dir, "/", dir_basename + "_splited_saved_", timestamp])
    exe = fluid.Executor(fluid.CPUPlace())

    print("[INFO] model_dir:{}".format(model_dir))

    # set params, model file names
    params_filename = ""
    model_filename = ""
    if os.path.exists(model_dir + "/" + model_name):
        model_filename = model_name
    else:
        print("[ERROR] model do not exist ")

    if os.path.exists(model_dir + "/" + param_name):
        params_filename = param_name
    else:
        print("[ERROR] params do not exist ")
    # elif os.path.exists(model_dir + "/params"):
    #     params_filename = "params"
    # else:
    #     print("[ERROR] No model parameters file `weights` or `params` found")
    #     exit(0)
    # if os.path.exists(model_dir + "/model"):
    #     model_filename = "model"
    # elif os.path.exists(model_dir + "/__model__"):
    #     model_filename = "__model__"
    # else:
    #     print("[ERROR] No model file `model` or `__model__` found")
    #     exit(0)
    print("[INFO] model_filename:{}".format(model_filename))
    print("[INFO] params_filename:{}".format(params_filename))

    # load model
    [inference_program, feed_target_names, fetch_targets] = \
      fluid.io.load_inference_model(dirname=model_dir, executor=exe, model_filename=model_filename, params_filename=params_filename)

    # save params as split files
    os.mkdir(save_model_dir)
    fluid.io.save_persistables(
        exe, save_model_dir, main_program=inference_program)
    print("[INFO] save split params file to {}".format(save_model_dir))

    # copy model to same directory
    model_src_path = "".join([model_dir, "/", model_filename])
    model_dst_path = "".join([save_model_dir, "/", "__model__"])
    shutil.copyfile(model_src_path, model_dst_path)
    if model_filename == "model":
        shutil.copyfile(model_src_path, "".join([save_model_dir,
                                                 "/__model__"]))
    print(
        "[INFO] rename source `model` file as `__model__` and copy from old path:{} to new path:{}"
        .format(model_src_path, model_dst_path))
