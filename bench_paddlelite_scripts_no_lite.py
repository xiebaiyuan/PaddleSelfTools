# %% [markdown]
# ### Create Dir IF NOT EXISTS

# %%
import argparse
import sys
import numpy as np
import os
import paddle.fluid as fluid
from paddle.fluid import debugger
from paddle.fluid import core
import subprocess
import paddle

paddle.enable_static()


def create_dir_if_not_exist(dir_path):
    import os
    if dir_path != "" and not os.path.exists(dir_path):
        os.makedirs(dir_path)


# %% [markdown]
# ### SHELL TOOLS
# %%
import subprocess


def sh(command):
    pipe = subprocess.Popen(command,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    outputs = ""
    while True:
        line = pipe.stdout.readline()
        if not line:
            break
        # print(line.decode("utf-8"))
        outputs += line.decode("utf-8")
    return outputs


def load_inference_model(model_path, model_filename, params_filename):
    exe = fluid.Executor(fluid.CPUPlace())
    net_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(dirname=model_path, executor=exe,
                                                                                  model_filename=model_filename,
                                                                                  params_filename=params_filename)
    return exe, net_program, feed_target_names, fetch_targets


# ### COMMON TOOLS
def fetch_tmp_vars(block, fetch_targets, var_names_list=None):
    """
    """

    def var_names_of_fetch(fetch_targets):
        var_names_list = []
        for var in fetch_targets:
            var_names_list.append(var.name)
        return var_names_list

    fetch_var = block.var('fetch')
    old_fetch_names = var_names_of_fetch(fetch_targets)
    new_fetch_vars = []
    for var_name in old_fetch_names:
        var = block.var(var_name)
        new_fetch_vars.append(var)
    i = len(new_fetch_vars)
    if var_names_list is None:
        var_names_list = block.vars.keys()
    for var_name in var_names_list:
        if var_name != '' and var_name not in old_fetch_names:
            var = block.var(var_name)
            new_fetch_vars.append(var)
            block.append_op(
                type='fetch',
                inputs={'X': [var_name]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})
            i = i + 1
    return new_fetch_vars


def run_and_pull_paddlelite_result(model_path, model_filename, params_filename):
    # use opt to get fused graph and tmp var names
    # sh(nomemoptimized_opt + " --model_dir=" + model_type + " --optimize_out_type=protobuf --valid_targets=arm --optimize_out=./" + model_type + "/opted/")

    ## origin model
    exe, net_program, feed_target_names, fetch_targets = load_inference_model(model_path, model_filename,
                                                                              params_filename)

    # ## opt model
    # exe_opted, net_program_opted, feed_target_names_opted, fetch_targets_opted = load_inference_model(
    #     model_path + "/opted", "model", "params")

    # ### GET VAR SHAPES
    def get_var_shape(var_name):
        vars = net_program.current_block().vars
        shape = vars[var_name].desc.shape()
        for i in range(len(shape)):
            dim = shape[i]
            if dim == -1:
                shape[i] = 1
        return shape

    def get_var_shape_str(shape):
        return "" + ",".join([str(x) for x in shape]) + ""

    ## lite runnable feed str generate
    def get_feed_shape_str():
        shapes_str = ""
        for name in feed_target_names:
            shape_str = get_var_shape_str(get_var_shape(name))
        shapes_str += shape_str;
        shapes_str += ":";
        return shapes_str

    for name in feed_target_names:
        # print(name)
        shape = get_var_shape(name)
        # print(get_var_shape_str(shape))

    feed_shape_str = get_feed_shape_str()
    # print(feed_shape_str)

    origin_fetch_names = []
    for fetch_target in fetch_targets:
        # print(name)
        origin_fetch_names.append(fetch_target.name.replace("/", "_"))

    ops = net_program.current_block().ops
    vars = net_program.current_block().vars

    addition_tmp_vars = []
    for op in ops:
        for var_name in op.output_arg_names:
            if var_name == "fetch":
                continue
            var = vars[var_name]
            if not var.persistable:
                # put non-presistable output var into tmp list
                addition_tmp_vars.append(var_name)

    # ### GENERATE TEST MODELS
    global_block = net_program.global_block()
    fetch_targets = fetch_tmp_vars(global_block, fetch_targets, addition_tmp_vars)

    test_model_path = model_path + "/" + model_path + "_test_model"
    create_dir_if_not_exist(test_model_path)
    fluid.io.save_inference_model(dirname=test_model_path,
                                  feeded_var_names=feed_target_names,
                                  target_vars=fetch_targets,
                                  executor=exe,
                                  main_program=net_program,
                                  model_filename=model_filename,
                                  params_filename=params_filename)

    # # 将测试模型转化为 PaddleLite 可执行的.nb文件, 目标路径是execu
    # cmd_opt_test_model = nomemoptimized_opt + " --model_dir=" + test_model_path + " --optimize_out_type=naive_buffer --valid_targets=arm --optimize_out=" + execuable_path + "/" + model_type + "_test_model_arm"
    # # print(cmd_opt_test_model)
    # sh(cmd_opt_test_model)

    # # push models and exe to mobile , run and pull results.

    # # sh("cp -r test_model_path {0}/models/cpu/;cp -r test_model_path {0}/models/gpu/ ".format(mml_native_path))
    # sh("adb shell rm -rf /data/local/tmp/workbenchtest")
    # sh("adb shell mkdir /data/local/tmp/workbenchtest")
    # sh("adb push " + execuable_path + "/* /data/local/tmp/workbenchtest")
    # exe_cmd = "adb shell \"cd /data/local/tmp/workbenchtest &&  export LD_LIBRARY_PATH=. && chmod +x " \
    #           "./test_mml_models && ./test_mml_models ./" + model_type + "_test_model_arm.nb " + feed_shape_str + " 1 0 0 0 \" "
    # sh(exe_cmd)
    # mobile_result_path = model_type + "_test_model_arm.nb_vars"
    # sh("rm -rf " + mobile_result_path)
    # sh("adb pull /data/local/tmp/workbenchtest/" + mobile_result_path + " " + model_path)


# need adb install on your mac

model_path = "./ar_cheji_day"
model_filename = "model"
params_filename = "params"
# execuable_path = "./modelrunner"
# nomemoptimized_opt = "./opt_mac"
run_and_pull_paddlelite_result(model_path, model_filename, params_filename)
