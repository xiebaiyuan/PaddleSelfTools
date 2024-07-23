#!/usr/bin/python3

import sys
import os
import time
import paddle
from paddleslim.analysis import flops


def get_dir_size(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum(
            [os.path.getsize(os.path.join(root, name)) for name in files])
    return size


def load_inference_model(model_path, exe):
    """
    Load the inference model and execute it.
    """
    return paddle.static.load_inference_model(model_path, exe)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ./{} <model_dir>".format(
            os.path.basename(__file__)))
        exit(0)

    # input parameters
    model_dir = sys.argv[1]  # whole dir
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    save_model_dir = "".join([model_dir, "/", "saved-", timestamp])
    paddle.set_device('cpu')

    print("[INFO] model_dir:{}".format(model_dir))

    paddle.enable_static()

    """
    Perform inference test with support for dynamic shapes.
    """
    exe = paddle.static.Executor(paddle.CPUPlace())

    [net_program, feed_target_names, fetch_targets] = load_inference_model(model_dir + "/model", exe)

    print(net_program)
    global_block = net_program.global_block()

    flops_value = flops(net_program)

    op_num = 0
    for block in net_program.blocks:
        op_num += len(block.ops)

    print("FLOPs: {:.6e} {:.3f}Mb  ops: {}".format(
        flops_value,
        get_dir_size(model_dir) / 1024 / 1024, op_num))
