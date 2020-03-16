# -*- coding: utf-8 -*
# %%
import os
import sys
import math
import subprocess
import numpy as np
import paddle.fluid as fluid
IS_DEBUG = False
is_sample_step = False
sample_step = 100
sample_num = 20

need_save = True
diff_threshold = 0.1
feed_all_1 = False
force_gen_inputs_outputs = False
need_print_mean = True
show_correct_check = False
need_check_mobile = False

need_wanted = False
wanted_list = [
    "blocks.2.0.se.conv_reduce.tmp_2", "blocks.2.0.se.conv_reduce.tmp_1",
    "blocks.2.0.se.conv_reduce.tmp_0"
]
model_name = "lens_mnasnet"
# model_name = "performancemodelv3"
# model_name = "lens_nanoyolo"
model_path = "/data/coremodels/" + model_name + "/"

checked_model_path = model_path + "/" + "checked_model"
feed_path = model_path + "/" + "feeds"
output_path = model_path + "/" + "outputs"

mobile_exec_root = "/data/local/tmp/bin"
#######  LITE SOURCE CONFIG
need_check_model_nb = False
lite_exec_root = "/data/local/tmp/opencl"
# test_name = "test_nanoyolo"

lite_push_model_dir = "{}/models/{}/".format(lite_exec_root, model_name)
# push_model_dir = "/data/local/tmp/opencl/models/nanoyolo/"
# input_name = "image"
# output_name = "save_infer_model_scale_0"
split_dir = "split_model"
# split_dir = "saved-20200302-164315"
lite_source_model_dir = "/data/coremodels/{}/{}/".format(model_name, split_dir)

lite_src_root = os.path.abspath("./") + "/"
if IS_DEBUG:
    print("lite_src_root :{}".format(lite_src_root))
# if lite_src_root.endswith("/"):
#     lite_src_root = lite_src_root[:-1]
#######

###### mobile config ######
is_lod = False
mobile_model_path = ""
fast_check = False

need_encrypt = False
check_exception = False
checked_encrypt_model_path = "checked_encrypt_model"
output_var_filter = []
output_key_filter = {}
check_shape = False
quantification = False
quantification_fold = 100000
architecture = "arm-v7a"
# architecture = "arm-v8a"
correct_persistable = False
###### mobile config end ######

np.set_printoptions(linewidth=150)

feed_names_ = []

mobile_src_root = os.path.abspath("../../../")
if mobile_src_root.endswith("/"):
    mobile_src_root = mobile_src_root[:-1]

dot = "•"


def black(x):
    return "\033[30m" + str(x) + "\033[0m"


def red(x):
    return "\033[31m" + str(x) + "\033[0m"


def green(x):
    return "\033[32m" + str(x) + "\033[0m"


def yellow(x):
    return "\033[33m" + str(x) + "\033[0m"


def reset(x):
    return "\033[0m" + str(x)


# %%
def print_e(e):
    if not check_exception:
        return
    print('str(Exception):\t', str(Exception))
    print('str(e):\t\t', str(e))
    print('repr(e):\t', repr(e))


# %%
def pp_tab(x, level=0):
    header = ""
    for i in range(0, level):
        header += "\t"
    print(header + str(x), flush=True)


def pp_black(x, level=0):
    pp_tab(black(x) + reset(""), level)


def pp_red(x, level=0):
    pp_tab(red(x) + reset(""), level)


def pp_green(x, level=0):
    pp_tab(green(x) + reset(""), level)


def pp_yellow(x, level=0):
    pp_tab(yellow(x) + reset(""), level)


def sh(command):
    pipe = subprocess.Popen(command,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return pipe.stdout.read().decode("utf-8")


def push(src, dest=""):
    sh("adb push {} {}".format(src, mobile_exec_root + "/" + dest))


def push_lite(src, dest=""):
    if IS_DEBUG:
        pp_yellow("push{}".format(src))
    result = sh("adb push {} {}".format(src, lite_exec_root + "/" + dest))
    if result.find("adb: error") != -1:
        pp_red("adb push err: {}".format(result))
    if IS_DEBUG:
        pp_red("{}".format(result))


pp_yellow(dot + " start inspecting fluid model")

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())


# %%
# 加载模型
def load_model(model_path):
    prog, feeds, fetches = fluid.io.load_inference_model(
        dirname=model_path,
        executor=exe,
        model_filename="model",
        params_filename="params")
    global correct_persistable
    if correct_persistable:
        ops = prog.current_block().ops
        vars = prog.current_block().vars
        for op in ops:
            for var_name in op.output_arg_names:
                if var_name == "fetch":
                    continue
                var = vars[var_name]
                if var.persistable:
                    pp_red("has found non-persistable output var : {}".format(
                        var_name))
                    var.persistable = False
    return (prog, feeds, fetches)


prog, feeds, fetches = load_model(model_path)


# %%
# 强制要求所有张量的形状，在model和params中一致，并重新保存模型
def resave_model(feed_kv):
    if len(mobile_model_path) > 0:
        pp_green("has set mobile_model_path, stop checking model & params", 1)
        sh("cp {}/* {}".format(mobile_model_path, checked_model_path))
        return
    ops = prog.current_block().ops
    vars = prog.current_block().vars
    # 强制所有var为可持久化
    p_names = []
    for name in vars:
        name = str(name)
        v = fluid.framework._get_var(name, prog)
        if not v.persistable:
            v.persistable = True
            p_names.append(name)
    outputs = run_model(feed_kv=feed_kv)
    has_found_wrong_shape = False
    # 修正每个var的形状
    for name in vars:
        name = str(name)
        v = vars[name]
        if v.persistable:
            v1 = fluid.global_scope().find_var(name)
            try:
                t1 = v1.get_tensor()
                shape = t1.shape()
            except Exception as e:
                print_e(e)
                continue
            if v.desc.shape() != shape:
                has_found_wrong_shape = True
            v.desc.set_shape(shape)
    # 恢复var的可持久化属性
    for name in p_names:
        v = fluid.framework._get_var(name, prog)
        v.persistable = False
    if not quantification:
        fluid.io.save_inference_model(dirname=checked_model_path,
                                      feeded_var_names=feeds,
                                      target_vars=fetches,
                                      executor=exe,
                                      main_program=prog,
                                      model_filename="model",
                                      params_filename="params")
    if has_found_wrong_shape:
        pp_red("has found wrong shape", 1)
    else:
        pp_green("has not found wrong shape", 1)
    pp_green(
        "new model is saved into directory 【{}】".format(checked_model_path), 1)


# 分别加密model和params，加密key使用同一个


# %%
def encrypt_model():
    if not need_encrypt:
        return
    pp_yellow(dot + dot + " encrypting model")
    if not os.path.exists(checked_encrypt_model_path):
        os.mkdir(checked_encrypt_model_path)
    res = sh("model-encrypt-tool/enc_key_gen -l 20 -c 232")
    lines = res.split("\n")

    for line in lines:
        if line.startswith("key:"):
            line = line.replace('key:', '')
            sh("model-encrypt-tool/enc_model_gen -k '{}' -c 2 -i checked_model/model -o "
               "checked_model/model.ml".format(line))
            sh("model-encrypt-tool/enc_model_gen -k '{}' -c 2 -i checked_model/params  -o checked_model/params.ml"
               .format(line))
            pp_green("model has been encrypted, key is : {}".format(line), 1)
            sh("mv {} {}".format(checked_model_path + "/*.ml",
                                 checked_encrypt_model_path))
            return
    pp_red("model encrypt error", 1)


# %%
# 生成feed的key-value对
def gen_feed_kv():
    feed_kv = {}
    for feed_name in feeds:
        feed_shape = get_feed_var_shape(feed_name)
        data = np.random.random(feed_shape).astype("float32")
        feed_kv[feed_name] = data
        if feed_all_1:
            feed_kv[feed_name] = np.ones(feed_shape).astype("float32")
    return feed_kv


# %%
# 保存feed的key-value对
def save_feed_kv(feed_kv):
    for feed_name in feed_kv:
        feed_data = feed_kv[feed_name]
        feed_list = feed_data.flatten().tolist()
        if not os.path.exists(feed_path):
            os.mkdir(feed_path)
        file_name = feed_name.replace("/", "_")
        out_file = open(feed_path + "/" + file_name, "w")
        for feed_item in feed_list:
            out_file.write("{}\n".format(feed_item))
        out_file.close()


# %%
last_feed_var_name = None
last_feed_file_name = None
last_feed_var_lod = None
last_fetch_var_name = None


# 加载feed的key-value对
def load_feed_kv():
    if not os.path.exists(feed_path):
        return None
    global last_feed_var_name
    global last_feed_file_name
    global last_feed_var_lod
    feed_kv = {}
    pp_yellow(dot + dot + " checking feed info")
    pp_green("feed data is saved into directory 【{}】".format(feed_path), 1)
    for feed_name in feeds:
        feed_shape = get_feed_var_shape(feed_name)
        pp_tab(
            "feed var name : {}; feed var shape : {}".format(
                feed_name, feed_shape), 1)
        file_name = feed_name.replace("/", "_")
        last_feed_var_name = feed_name
        last_feed_file_name = file_name
        feed_file_path = feed_path + "/" + file_name
        if not os.path.exists(feed_file_path):
            return None
        data = np.loadtxt(feed_file_path)
        expected_len = 1
        for dim in feed_shape:
            expected_len *= dim
        if len(np.atleast_1d(data)) != expected_len:
            return None
        data = data.reshape(feed_shape).astype("float32")

        if is_lod:
            data_shape = [1]
            for dim in feed_shape:
                data_shape.append(dim)
            data = data.reshape(data_shape).astype("float32")
            tensor = fluid.LoDTensor()
            seq_lens = [len(seq) for seq in data]
            cur_len = 0
            lod = [cur_len]
            for l in seq_lens:
                cur_len += l
                lod.append(cur_len)
            data = data.reshape(feed_shape)
            tensor.set(data, fluid.CPUPlace())
            tensor.set_lod([lod])
            last_feed_var_lod = lod
            feed_kv[feed_name] = tensor
        else:
            feed_kv[feed_name] = data
    return feed_kv


# %%
# 运行模型
def run_model(feed_kv=None):
    if feed_kv is None:
        feed_kv = gen_feed_kv()
    outputs = exe.run(prog,
                      feed=feed_kv,
                      fetch_list=fetches,
                      return_numpy=False)
    feed_names_.clear()
    for feed_name in feeds:
        feed_names_.append(feed_name)
        # pp_green(feed_name, 1)
    results = []
    for output in outputs:
        results.append(np.array(output))
    return results


# %%
# 获取变量形状
def get_var_shape(var_name):
    vars = prog.current_block().vars
    shape = vars[var_name].desc.shape()
    for i in range(len(shape)):
        dim = shape[i]
        if dim == -1:
            shape[i] = 1
    return shape


# %%
# 获取输入变量形状
def get_feed_var_shape(var_name):
    # 如果想写死输入形状，放开以下语句
    # return [1, 3, 224, 224]
    return get_var_shape(var_name)


# %%
persistable_cache = []


# 所有var，全部变成持久化
def force_all_vars_to_persistable():
    global persistable_cache
    for var_name in vars.keys():
        var_name = str(var_name)
        v = fluid.framework._get_var(var_name, prog)
        persistable = v.persistable
        if not persistable:
            persistable_cache.append(var_name)
            v.persistable = True


# %%
# 恢复持久化属性
def restore_all_vars_persistable():
    global persistable_cache
    for var_name in vars.keys():
        var_name = str(var_name)
        v = fluid.framework._get_var(var_name, prog)
        persistable = v.persistable
        if var_name in persistable_cache:
            v.persistable = False
    persistable_cache = []


# %%
# 获取var的数据
def get_var_data(var_name, feed_kv=None):
    output = np.array(fluid.global_scope().var(var_name).get_tensor())
    return output


# %%
output_var_cache = {}


def tensor_sample(tensor):
    if is_sample_step:
        step = sample_step
    else:
        step = math.floor(len(tensor) / sample_num)
    step = max(step, 1)
    step = int(step)
    sample = []
    for i in range(0, len(tensor), step):
        sample.append(tensor[i])
    return sample


# %%
#计算mean值
mean_dict = {}


def calc_mean(name, tensor):
    mean = -1
    try:
        step = 1
        step = int(step)
        sum = 0.0
        for i in range(0, len(tensor), step):
            sum += tensor[i]
        mean = sum / len(tensor)
        pp_green(
            "{0:30}  {1:30.5f}     {2:30}".format(name, mean,
                                                  str(get_var_shape(name))), 2)
        mean_dict[name] = mean

    except Exception as e:
        print_e(e)

    return mean


# %%
op_cache = {}


# %%
# 获取每层输出的数据
def save_all_op_output(feed_kv=None):
    force_all_vars_to_persistable()
    outputs = run_model(feed_kv=feed_kv)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    ops = prog.current_block().ops
    fetch_names = []
    for fetch in fetches:
        fetch_names.append(fetch.name)
    feed_names = feeds
    if len(output_var_filter) > 0:
        for fetch_name in fetch_names:
            output_var_filter.append(fetch_name)
    for i in range(len(ops)):
        op = ops[i]
        var_name = None
        var_name_index = -1
        for index in range(len(op.output_names)):
            if op.output_names[index] in ["Y", "Out", "Output"]:
                var_name_index = index
                break
        if var_name_index != -1:
            var_name = op.output_arg_names[var_name_index]
        else:
            for name in op.output_arg_names:
                var_name = name
                if "tmp" in name:
                    break
        if len(output_var_filter) > 0:
            if var_name not in output_var_filter:
                continue
        # real_var_name = None
        # if op.type == "fetch":
        #     for name in op.input_arg_names:
        #         real_var_name = name
        #         if "tmp" in name:
        #             break
        # else:
        #     real_var_name = var_name
        if fast_check:
            if var_name not in fetch_names and var_name not in feed_names:
                continue
        try:
            data = get_var_data(var_name, feed_kv=feed_kv).flatten().tolist()
            sample = tensor_sample(data)
            # 计算均值
            if need_print_mean:
                calc_mean(var_name, data)
            output_var_cache[var_name] = (sample)
            op_cache[i] = (var_name, op)
            file_name = var_name.replace("/", "_")
            if need_save or force_gen_inputs_outputs:
                out_file = open(output_path + "/" + file_name, "w")
                if var_name in feed_names:
                    for item in data:
                        out_file.write("{}\n".format(item))
                else:
                    for item in sample:
                        out_file.write("{}\n".format(item))
                out_file.close()
            else:
                pass
        except Exception as e:
            print_e(e)
    for i in range(len(ops)):
        op = ops[i]
        if op.type not in output_key_filter:
            continue
        var_name = None
        var_name_index = -1
        for index in range(len(op.output_names)):
            if op.output_names[index] in output_key_filter[op.type]:
                var_name_index = index
                break
        if var_name_index != -1:
            var_name = op.output_arg_names[var_name_index]
        else:
            continue
        if len(output_var_filter) > 0:
            if var_name not in output_var_filter:
                continue
        # real_var_name = None
        # if op.type == "fetch":
        #     for name in op.input_arg_names:
        #         real_var_name = name
        #         if "tmp" in name:
        #             break
        # else:
        #     real_var_name = var_name
        if fast_check:
            if var_name not in fetch_names and var_name not in feed_names:
                continue
        try:
            data = get_var_data(var_name, feed_kv=feed_kv).flatten().tolist()
            sample = tensor_sample(data)
            output_var_cache[var_name] = (sample)
            op_cache[i] = (var_name, op)
            file_name = var_name.replace("/", "_")
            out_file = open(output_path + "/" + file_name, "w")
            if var_name in feed_names:
                pp_green("loop 1-  write : {}".format(var_name), 2)
                for item in data:
                    out_file.write("{}\n".format(item))
            else:
                pp_green("loop 1-  write : {}".format(var_name), 2)
                for item in sample:
                    out_file.write("{}\n".format(item))
            out_file.close()
        except Exception as e:
            print_e(e)

    restore_all_vars_persistable()
    force_all_vars_to_persistable()
    outputs = run_model(feed_kv=feed_kv)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    ops = prog.current_block().ops
    fetch_names = []
    for fetch in fetches:
        fetch_names.append(fetch.name)
    feed_names = feeds
    if len(output_var_filter) > 0:
        for fetch_name in fetch_names:
            output_var_filter.append(fetch_name)
    for i in range(len(ops)):
        op = ops[i]
        var_name = None
        var_name_index = -1
        for index in range(len(op.output_names)):
            if op.output_names[index] in ["Y", "Out", "Output"]:
                var_name_index = index
                break
        if var_name_index != -1:
            var_name = op.output_arg_names[var_name_index]
        else:
            for name in op.output_arg_names:
                var_name = name
                if "tmp" in name:
                    break
        if len(output_var_filter) > 0:
            if var_name not in output_var_filter:
                continue
        # real_var_name = None
        # if op.type == "fetch":
        #     for name in op.input_arg_names:
        #         real_var_name = name
        #         if "tmp" in name:
        #             break
        # else:
        #     real_var_name = var_name
        if fast_check:
            if var_name not in fetch_names and var_name not in feed_names:
                continue
        try:
            data = get_var_data(var_name, feed_kv=feed_kv).flatten().tolist()
            sample = tensor_sample(data)
            output_var_cache[var_name] = (sample)
            op_cache[i] = (var_name, op)
            file_name = var_name.replace("/", "_")
            if need_save or force_gen_inputs_outputs:
                out_file = open(output_path + "/" + file_name, "w")
                if var_name in feed_names:
                    # pp_green("loop 2-  write : {}".format(var_name), 2)
                    for item in data:
                        out_file.write("{}\n".format(item))
                else:
                    # pp_green("loop 2-  write : {}".format(var_name), 2)
                    for item in sample:
                        out_file.write("{}\n".format(item))
                out_file.close()
        except Exception as e:
            print_e(e)
    pp_green(
        "all the op outputs are saved into directory 【{}】".format(output_path),
        1)
    for i in range(len(ops)):
        op = ops[i]
        if op.type not in output_key_filter:
            continue
        var_name = None
        var_name_index = -1
        for index in range(len(op.output_names)):
            if op.output_names[index] in output_key_filter[op.type]:
                var_name_index = index
                break
        if var_name_index != -1:
            var_name = op.output_arg_names[var_name_index]
        else:
            continue
        if len(output_var_filter) > 0:
            if var_name not in output_var_filter:
                continue
        # real_var_name = None
        # if op.type == "fetch":
        #     for name in op.input_arg_names:
        #         real_var_name = name
        #         if "tmp" in name:
        #             break
        # else:
        #     real_var_name = var_name
        if fast_check:
            if var_name not in fetch_names and var_name not in feed_names:
                continue
        try:
            data = get_var_data(var_name, feed_kv=feed_kv).flatten().tolist()
            pp_yellow("var_name : [{}]".format(output_path), 1)
            sample = tensor_sample(data)
            output_var_cache[var_name] = (sample)
            op_cache[i] = (var_name, op)
            file_name = var_name.replace("/", "_")
            out_file = open(output_path + "/" + file_name, "w")
            if var_name in feed_names:
                pp_green("loop 3-  write : {}".format(var_name), 2)
                for item in data:
                    out_file.write("{}\n".format(item))
            else:
                pp_green("loop 3-  write : {}".format(var_name), 2)
                for item in sample:
                    out_file.write("{}\n".format(item))
            out_file.close()
        except Exception as e:
            print_e(e)

    restore_all_vars_persistable()


ops = prog.current_block().ops
vars = prog.current_block().vars

pp_yellow(dot + dot + " checking op list")
op_types = set()
for op in ops:
    op_types.add(op.type)
pp_tab("op types : {}".format(op_types), 1)


#%%
def save_lite_inputs(lines):
    # print("save lite inputs")
    input_cache = {}
    for line in lines:
        parts = line.split(" ")

        if len(parts) < 2:
            continue
        if "lite-auto-test-input" != parts[0]:
            continue

        # if parts[1] == "load-time-cost":
        #     pp_green("load time cost : {}".format(parts[2]), 1)
        # elif parts[1] == "predict-time-cost":
        #     pp_green("predict time cost : {}".format(parts[2]), 1)
        # elif parts[1] == "preprocess-time-cost":
        #     pp_green("preprocess time cost : {}".format(parts[2]), 1)
        elif parts[1] == "var":
            # print(str(parts))
            var_name = parts[2]
            values = list(map(lambda x: float(x), parts[3:]))
            input_cache[var_name] = values
        for name in input_cache:
            print(name)
            out_file = open(feed_path + "/" + "image_recheck" + name, "w")
            for v in input_cache[name]:
                out_file.write("{}\n".format(v))
            out_file.close()
        # try:
        #     if not os.path.exists("feed"):
        #         os.mkdir(feed_path)
        # file_name = feed_name.replace("/", "_")
        # out_file = open(feed_path + "/" + file_n
        # me, "w")
        # for feed_item in feed_list:
        #     out_file.write("{}\n".format(feed_item))
        # out_file.close()
        # except Exception as e:
        #     print_e(e)


# %%
def check_mobile_results(args, fuse, mem_opt):
    args = "{} {} {} {} {}".format("1" if fuse else "0",
                                   "1" if mem_opt else "0",
                                   "1" if quantification else "0",
                                   quantification_fold, args)
    res = sh(
        "adb shell \"cd {} && export LD_LIBRARY_PATH=. && ./test-net {}\"".
        format(mobile_exec_root, args))
    lines = res.split("\n")
    # for line in lines:
    #     print(line)

    for line in lines:
        if line.startswith("auto-test-debug"):
            print(line)

    for line in lines:
        if line.startswith("mean :"):
            print(line)
    pp_yellow(dot + dot +
              " checking paddle mobile results for {} -- {} ".format(
                  green("【fusion】" if fuse else "【non fusion】"),
                  green("【memory-optimization】"
                        if mem_opt else "【non-memory-optimization】")))
    mobile_var_cache = {}
    for line in lines:
        parts = line.split(" ")
        if len(parts) < 2:
            continue
        if "auto-test" != parts[0]:
            continue
        if parts[1] == "load-time-cost":
            pp_green("load time cost : {}".format(parts[2]), 1)
        elif parts[1] == "predict-time-cost":
            pp_green("predict time cost : {}".format(parts[2]), 1)
        elif parts[1] == "preprocess-time-cost":
            pp_green("preprocess time cost : {}".format(parts[2]), 1)
        elif parts[1] == "var":
            var_name = parts[2]
            values = list(map(lambda x: float(x), parts[3:]))
            mobile_var_cache[var_name] = values
    error_index = None
    error_values1 = None
    error_values2 = None
    checked_names = []
    fetch_names = []
    for fetch in fetches:
        fetch_names.append(fetch.name)
    fetch_diff = 0.0
    fetch_count = 0
    for index in op_cache:
        op_output_var_name, op = op_cache[index]
        if not op_output_var_name in output_var_cache:
            continue
        if not op_output_var_name in mobile_var_cache:
            continue
        if op_output_var_name not in fetch_names:
            continue
        values1 = output_var_cache[op_output_var_name]
        values2 = mobile_var_cache[op_output_var_name]
        shape = get_var_shape(op_output_var_name) if check_shape else []
        for i in range(len(values1)):
            try:
                v1 = values1[i]
                v2 = values2[len(shape) + i]
                fetch_diff += abs(v1 - v2)
                fetch_count += 1
            except Exception as e:
                print_e(e)

    if fetch_count != 0:
        pp_yellow("output avg diff : {}".format(fetch_diff / fetch_count), 1)
    for index in op_cache:
        op_output_var_name, op = op_cache[index]
        if mem_opt:
            found_in_fetch = False
            for fetch in fetches:
                if op_output_var_name == fetch.name:
                    found_in_fetch = True
                    break
            if not found_in_fetch:
                continue
        if not op_output_var_name in output_var_cache:
            continue
        if not op_output_var_name in mobile_var_cache:
            continue
        if op_output_var_name not in fetch_names:
            continue
        values1 = output_var_cache[op_output_var_name]
        values2 = mobile_var_cache[op_output_var_name]
        shape = get_var_shape(op_output_var_name) if check_shape else []
        if len(values1) + len(shape) != len(values2):
            error_index = index
        for i in range(len(shape)):
            v1 = shape[i]
            v2 = values2[i]
            if v1 != v2:
                error_index = index
                break
        if error_index == None:
            for i in range(len(values1)):
                v1 = values1[i]
                v2 = values2[len(shape) + i]
                if abs(v1 - v2) > diff_threshold:
                    error_index = index
                    break
        checked_names.append(op_output_var_name)
        if error_index != None:
            error_values1 = values1
            error_values2 = values2
            break
    if error_index == None:
        for name in fetch_names:
            if name not in checked_names:
                error_index = -1
                break
    if error_index == None:
        pp_green("outputs are all correct", 1)
    elif error_index == -1:
        pp_red("outputs are missing")
    else:
        error_values1 = np.array(error_values1)
        error_values2 = np.array(error_values2)
        # pp_red("mobile op is not correct, error occurs at {}th op, op's type is {}")
        pp_red("outputs are incorrect", 1)
        pp_red("fluid results are : ", 1)
        pp_red(str(error_values1).replace("\n", "\n" + "\t" * 1), 1)
        pp_yellow("paddle mobile results are : ", 1)
        pp_red(str(error_values2).replace("\n", "\n" + "\t" * 1), 1)
        if not fuse and not mem_opt:
            pp_yellow("checking individual ops : ", 1)
            error_index = None
            error_values1 = None
            error_values2 = None
            checked_names = []
            fetch_names = []
            for fetch in fetches:
                fetch_names.append(fetch.name)
            for index in op_cache:
                op_output_var_name, op = op_cache[index]
                if mem_opt:
                    found_in_fetch = False
                    for fetch in fetches:
                        if op_output_var_name == fetch.name:
                            found_in_fetch = True
                            break
                    if not found_in_fetch:
                        continue
                if not op_output_var_name in output_var_cache:
                    continue
                if not op_output_var_name in mobile_var_cache:
                    continue
                if fuse or mem_opt:
                    if op_output_var_name not in fetch_names:
                        continue
                values1 = output_var_cache[op_output_var_name]
                values2 = mobile_var_cache[op_output_var_name]
                shape = get_var_shape(
                    op_output_var_name) if check_shape else []
                if len(values1) + len(shape) != len(values2):
                    error_index = index
                for i in range(len(shape)):
                    v1 = shape[i]
                    v2 = values2[i]
                    if v1 != v2:
                        error_index = index
                        break
                if error_index == None:
                    for i in range(len(values1)):
                        v1 = values1[i]
                        v2 = values2[len(shape) + i]
                        if ((not math.isnan(v1)) and math.isnan(v2)
                            ) or abs(v1 - v2) > diff_threshold:
                            error_index = index
                            pp_red(
                                "error:  index={0} {1:10.6f} > diff_threshold ---- {2:10.6f} - {3:10.6f} > {4:10.6f} "
                                .format(i, abs(v1 - v2), v1, v2,
                                        diff_threshold), 2)
                            # break
                checked_names.append(op_output_var_name)
                if error_index != None:
                    error_values1 = values1
                    error_values2 = values2
                    break
            if error_index == None:
                for name in fetch_names:
                    if name not in checked_names:
                        error_index = -1
                        break
            if error_index == None:
                pp_green("outputs are all correct", 1)
            elif error_index == -1:
                pp_red("outputs are missing")
            else:
                error_values1 = np.array(error_values1)
                error_values2 = np.array(error_values2)
                # pp_red("mobile op is not correct, error occurs at {}th op, op's type is {}")
                pp_red(
                    "corresponding fluid op is {}th op, op's type is {}, wrong var name is {}"
                    .format(error_index, op_cache[error_index][1].type,
                            op_output_var_name), 1)
                pp_red("fluid results are : ", 1)
                pp_red(str(error_values1).replace("\n", "\n" + "\t" * 1), 1)
                pp_yellow("paddle mobile results are : ", 1)
                pp_red(str(error_values2).replace("\n", "\n" + "\t" * 1), 1)
    # print(output_var_cache)
    # print(mobile_var_cache)


# %%


# 检查lite
def check_lite_results():
    print("")
    print("==================================================")
    print("")
    pp_yellow(dot + " start inspecting paddle lite correctness & performance")
    # input_name = last_feed_var_name
    # output_name = last_fetch_var_name
    test_name = "test_net_compare"

    pp_green(feed_names_, 1)
    feed_names_argu = ""
    input_des = ""
    output_des = ""

    vars = prog.current_block().vars
    pp_yellow(dot + dot + " 生成输入参数...")
    for n in feed_names_:
        feed_names_argu += "feed names: {}\n".format(n)
        pp_green("push : {} ".format(str(n)), 1)
        push_lite(feed_path + "/" + str(n), "{}".format(str(n)))
        input_des += n
        input_des += ":"
        shape = vars[n].desc.shape()
        for i in range(len(shape)):
            dim = shape[i]
            if dim == -1:
                shape[i] = 1
            input_des += str(dim)
            input_des += "_"
        input_des = input_des[:-1]
        input_des += ";"
    input_des = input_des[:-1]

    pp_yellow(dot + dot + " 生成输出参数...")
    for fn in fetches:
        n = fn.name
        push_key = n.replace("/", "_")
        feed_names_argu += "fetch names: {}\n".format(n)
        pp_green("push : {} ".format(str(n)), 1)
        push_lite(output_path + "/" + str(push_key),
                  "{}".format(str(push_key)))
        output_des += n
        output_des += ":"
        shape = vars[n].desc.shape()
        for i in range(len(shape)):
            dim = shape[i]
            if dim == -1:
                shape[i] = 1
            output_des += str(dim)
            output_des += "_"
        output_des = output_des[:-1]
        output_des += ";"
    output_des = output_des[:-1]
    if IS_DEBUG:
        pp_red("output_des: " + output_des)
    sh("adb shell mkdir -p /data/local/tmp/opencl")
    sh("adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/buffer")
    sh("adb shell mkdir -p /data/local/tmp/opencl/cl_kernel/image")
    # adb push lite/backends/opencl/cl_kernel/cl_common.h /data/local/tmp/opencl/cl_kernel/
    # adb push lite/backends/opencl/cl_kernel/buffer/* /data/local/tmp/opencl/cl_kernel/buffer/
    # adb push lite/backends/opencl/cl_kernel/image/* /data/local/tmp/opencl/cl_kernel/image/
    # no need to push cl kernels
    # push_lite(lite_src_root + "lite/backends/opencl/cl_kernel/cl_common.h",
    #           "cl_kernel/")
    # push_lite(lite_src_root + "lite/backends/opencl/cl_kernel/buffer/*",
    #           "cl_kernel/buffer/")
    # push_lite(lite_src_root + "lite/backends/opencl/cl_kernel/image/*",
    #           "cl_kernel/image/")

    sh("adb shell mkdir -p {}".format(lite_push_model_dir))
    # adb shell mkdir -p ${model_dir}
    # adb push ${input_dir}${input} /data/local/tmp/opencl/${input}
    # adb push ${output_dir}${output} /data/local/tmp/opencl/${output}
    # adb push ${source_model_dir}/* ${model_dir}
    remote_model_path = "models/{}/".format(model_name)
    local_nb_path_opencl = "{}/{}_opencl.nb".format(model_path, model_name)
    local_nb_path_arm = "{}/{}_arm.nb".format(model_path, model_name)

    push_lite(lite_source_model_dir + "/*", remote_model_path)

    if need_check_model_nb:
        push_lite(local_nb_path_opencl, remote_model_path + "/opencl.nb")
        push_lite(local_nb_path_arm, remote_model_path + "/arm.nb")

    push_lite(
        lite_src_root +
        "build.self.lite.android.armv7.gcc.opencl/lite/api/test_net_compare",
        "test_net_compare")
    # push_lite(feed_path + "/" + last_feed_file_name, last_feed_file_name)
    # "export GLOG_v=0; /data/local/tmp/opencl/${testname} --model_dir=${model_dir} --input_file=/data/local/tmp/opencl/${input} --output_file=/data/local/tmp/opencl/${output} --is_sample_step=false --sample_step=1 --sample_num=100 --checkscript=true --check_shape=false"
    pp_yellow(dot + dot + " 手机上执行推理...")
    exe_commend = "adb shell \"export GLOG_v=0; /data/local/tmp/opencl/{} --model_dir={} --input_file={} --output_file={} --is_sample_step={} --sample_step={} --sample_num={} --checkscript=true --check_shape={} --check_nb={}\"".format(
        test_name, lite_push_model_dir, input_des, output_des, is_sample_step,
        sample_step, sample_num, check_shape, need_check_model_nb)
    pp_yellow("\n{}  \n".format(exe_commend), 1)
    res = sh(exe_commend)
    if IS_DEBUG:
        print("执行推理成功>")
    lines = res.split("\n")

    if need_check_model_nb:
        pull_nb_commend = "adb pull /data/local/tmp/armoptmodel.nb {}".format(
            local_nb_path_arm)
        res = sh(pull_nb_commend)
        if IS_DEBUG:
            print(pull_nb_commend)
            print(res)

        pull_nb_commend = "adb pull /data/local/tmp/opencloptmodel.nb {}".format(
            local_nb_path_opencl)
        res = sh(pull_nb_commend)
        if IS_DEBUG:
            print(pull_nb_commend)
            print(res)
        pp_green("nb model is saved in {}  \n".format(model_path), 1)

    if IS_DEBUG:
        for line in lines:
            print(line)

    # for line in lines:
    #     if line.startswith("lite-auto-test"):
    #         print(line)

    # for line in lines:
    #     if line.startswith("mean :"):
    #         print(line)
    pp_yellow(dot + dot + " checking paddle lite results for {} -- {} ".format(
        last_feed_var_name, test_name))
    lite_var_cache = {}

    escape_list = []
    save_lite_inputs(lines)
    for line in lines:
        parts = line.split(" ")

        if len(parts) < 2:
            continue
        if "lite-auto-test" != parts[0]:
            continue

        # if parts[1] == "load-time-cost":
        #     pp_green("load time cost : {}".format(parts[2]), 1)
        # elif parts[1] == "predict-time-cost":
        #     pp_green("predict time cost : {}".format(parts[2]), 1)
        # elif parts[1] == "preprocess-time-cost":
        #     pp_green("preprocess time cost : {}".format(parts[2]), 1)
        elif parts[1] == "var":
            if parts[3] == "NOTGET":
                escape_list.append(parts[2])
                pass
            else:
                # print(str(parts))
                var_name = parts[2]
                values = list(map(lambda x: float(x), parts[3:]))
                lite_var_cache[var_name] = values
            # print(str(lite_var_cache))

    pp_green("skiped vars {} ".format(str(escape_list)), 1)
    error_index = None
    error_values1 = None
    error_values2 = None

    # checkfetch
    pp_yellow(dot + dot + " check fetchs results ...   ")
    checked_names = []
    fetch_names = []
    for fetch in fetches:
        fetch_names.append(fetch.name)
    fetch_diff = 0.0
    fetch_count = 0

    for index in op_cache:
        op_output_var_name, op = op_cache[index]
        if op_output_var_name in escape_list:
            # print("jump---{}".format(op_output_var_name))
            continue
        if not op_output_var_name in output_var_cache:
            continue
        if not op_output_var_name in lite_var_cache:
            continue
        if op_output_var_name not in fetch_names:
            continue
        # pp_green("{}:".format(op_output_var_name), 1)
        values1 = output_var_cache[op_output_var_name]
        values2 = lite_var_cache[op_output_var_name]
        shape = get_var_shape(op_output_var_name) if check_shape else []
        for i in range(len(values1)):
            v1 = values1[i]
            v2 = values2[len(shape) + i]
            fetch_diff += abs(v1 - v2)
            fetch_count += 1

    if fetch_count != 0:
        pp_yellow("output avg diff : {}".format(fetch_diff / fetch_count), 1)
    for index in op_cache:
        op_output_var_name, op = op_cache[index]
        # pp_green("pic {}----".format(op_output_var_name), 1)
        if True:
            found_in_fetch = False
            for fetch in fetches:
                if op_output_var_name == fetch.name:
                    found_in_fetch = True
                    break
            if not found_in_fetch:
                continue
        if op_output_var_name in escape_list:
            # print("jump-2--{}".format(op_output_var_name))
            continue
        if not op_output_var_name in output_var_cache:
            continue
        if not op_output_var_name in lite_var_cache:
            continue
        if op_output_var_name not in fetch_names:
            continue
        pp_green("check {}".format(op_output_var_name), 1)
        values1 = output_var_cache[op_output_var_name]
        values2 = lite_var_cache[op_output_var_name]
        shape = get_var_shape(op_output_var_name) if check_shape else []
        if len(values1) + len(shape) != len(values2):
            pp_red("{}: len not match".format(op_output_var_name))
            error_index = index
        for i in range(len(shape)):
            v1 = shape[i]
            v2 = values2[i]
            if v1 != v2:
                pp_red("shape not match  {} and {}".format(v1, v2))
                error_index = index
                break
        if error_index == None:
            for i in range(len(values1)):
                v1 = values1[i]
                v2 = values2[len(shape) + i]
                if abs(v1 - v2) > diff_threshold:
                    pp_red("value not match  {} and {}".format(v1, v2), 2)
                    error_index = index
                    break
        checked_names.append(op_output_var_name)
        if error_index != None:
            error_values1 = values1
            error_values2 = values2
            break

    # pp_green("fetch_names: {}".format(str(fetch_names)), 2)
    # pp_green("checked_names: {}".format(str(checked_names)), 2)
    if error_index == None:
        for name in fetch_names:
            if name not in checked_names:
                error_index = -1
                break
    if error_index == None:
        pp_green("outputs are all correct", 1)
    # elif error_index == -1:
    #     pp_red("outputs are missing")
    else:

        # check ops

        error_values1 = np.array(error_values1)
        error_values2 = np.array(error_values2)
        # pp_red("mobile op is not correct, error occurs at {}th op, op's type is {}")
        pp_red("outputs are incorrect", 1)
        pp_yellow("fluid results are : ", 1)
        pp_red(str(error_values1).replace("\n", "\n" + "\t" * 1), 1)
        pp_yellow("paddle lite results are : ", 1)
        pp_red(str(error_values2).replace("\n", "\n" + "\t" * 1), 1)
        pp_yellow(dot + dot + " check ops ...   ")

        pp_yellow("checking individual ops : ", 1)
        error_index = None
        error_values1 = None
        error_values2 = None
        checked_names = []
        fetch_names = []
        for fetch in fetches:
            fetch_names.append(fetch.name)
        pp_red("fetch_names:{}".format(str(fetch_names)), 1)
        for index in op_cache:
            op_output_var_name, op = op_cache[index]
            pp_green("check {}  ".format(op_output_var_name), 1)
            if (op_output_var_name not in wanted_list) and need_wanted:
                pp_green("not wanted skipped ", 2)
                continue
            if op_output_var_name in escape_list:
                pp_green("not get skipped ", 2)
                continue
            if not op_output_var_name in output_var_cache:
                pp_red("not in fluid output_var_cache skipped ", 2)
                continue
            if not op_output_var_name in lite_var_cache:
                pp_red("not in lite_var_cache skipped", 2)
                continue
            # if op_output_var_name not in fetch_names:
            #     pp_red("{}:not in fetch_names".format(op_output_var_name),
            #            2)
            #     continue

            values1 = output_var_cache[op_output_var_name]
            values2 = lite_var_cache[op_output_var_name]
            shape = get_var_shape(op_output_var_name) if check_shape else []
            if len(values1) + len(shape) != len(values2):
                pp_red(
                    "len(values1) + len(shape) {} and len(values2){} not match "
                    .format((len(values1) + len(shape)), len(values2)), 2)
                error_index = index
            else:
                if show_correct_check:
                    pp_green(
                        "correct len(values1) + len(shape) {} and len(values2){}  match "
                        .format((len(values1) + len(shape)), len(values2)), 2)

            for i in range(len(shape)):
                v1 = shape[i]
                v2 = values2[i]
                if v1 != v2:
                    pp_red("v1 != v2 ---- {} !={} not match ".format(v1, v2),
                           2)
                    error_index = index
                    break
                else:
                    if show_correct_check:
                        pp_green(
                            "correct shape1 == shape2 ---- {} !={}  match ".
                            format(v1, v2), 2)

            if error_index == None:
                for i in range(len(values1)):
                    v1 = values1[i]
                    v2 = values2[len(shape) + i]
                    diff = abs(v1 - v2)
                    if ((not math.isnan(v1))
                            and math.isnan(v2)) or diff > diff_threshold:
                        pp_red(
                            "error:  index={0} {1:10.6f} > diff_threshold ---- {2:10.6f} - {3:10.6f} > {4:10.6f} "
                            .format(i, diff, v1, v2, diff_threshold), 2)
                        error_index = index
                        # break
                    else:
                        if show_correct_check:
                            pp_green(
                                "correct : index={0}  {1:10.6f} < diff_threshold ---- {2:10.6f} - {3:10.6f} < {4:10.6f} "
                                .format(i, diff, v1, v2, diff_threshold), 2)

            checked_names.append(op_output_var_name)
            if error_index != None:
                error_values1 = values1
                error_values2 = values2
                break
        # 输出是否被检查?
        # if error_index == None:
        #     for name in fetch_names:
        #         if name not in checked_names:
        #             error_index = -1
        #             break
        if error_index == None:
            pp_green("outputs are all correct", 1)
        elif error_index == -1:
            pp_red("outputs are missing")
        else:
            error_values1 = np.array(error_values1)
            error_values2 = np.array(error_values2)
            # pp_red("mobile op is not correct, error occurs at {}th op, op's type is {}")
            pp_red(
                "corresponding fluid op is {}th op, op's type is {}, wrong var name is {}"
                .format(error_index, op_cache[error_index][1].type,
                        op_output_var_name), 1)
            pp_red("fluid results are : ", 1)
            pp_red(str(error_values1).replace("\n", "\n" + "\t" * 1), 1)
            pp_yellow("paddle lite results are : ", 1)
            pp_red(str(error_values2).replace("\n", "\n" + "\t" * 1), 1)
    # print(output_var_cache)
    # print(mobile_var_cache)
    result = sh(
        "cp {0}/lite/api/test_net_compare.cc {0}/PaddleMobileTools/lite/bk/".
        format(lite_src_root))
    if IS_DEBUG:
        print(result)


# 检查mobile
#%%
def check_mobile():
    if not need_check_mobile:
        return
    print("")
    print("==================================================")
    print("")
    pp_yellow(dot +
              " start inspecting paddle mobile correctness & performance")
    push(checked_model_path)
    push(feed_path + "/" + last_feed_file_name, "input.txt")
    push(mobile_src_root +
         "/build/release/{}/build/libpaddle-mobile.so".format(architecture))
    push(mobile_src_root +
         "/build/release/{}/build/cl_kernel".format(architecture))
    push(mobile_src_root + "/test/build/test-net")
    last_feed_var_shape = get_feed_var_shape(last_feed_var_name)
    args = str(len(last_feed_var_shape))
    for dim in last_feed_var_shape:
        args += " " + str(dim)
    if is_lod:
        args += " 1"
        args += " " + str(len(last_feed_var_lod))
        for dim in last_feed_var_lod:
            args += " " + str(dim)
    else:
        args += " 0"
    args += " " + str(len(output_var_cache))
    args += " " + str(1 if is_sample_step else 0)
    if is_sample_step:
        args += " " + str(sample_step)
    else:
        args += " " + str(sample_num)
    for var_name in output_var_cache.keys():
        args += " " + var_name
    args += " " + str(1 if check_shape else 0)
    # if not fast_check:
    check_mobile_results(args, False, False)
    #     check_mobile_results(args, False, True)
    # check_mobile_results(args, True, False)
    # check_mobile_results(args, True, True)


#%%
def gen_meanname_vectors(mean):
    keys = mean_dict.keys()
    vector_string = "std::vector<std::string> tensor_names = {"
    for key in keys:
        vector_string += "\"" + key + "\" ,"
    vector_string = vector_string[:-1]
    vector_string += "};"
    print(vector_string)


#%%
def main():
    # 加载kv
    feed_kv = load_feed_kv()
    if feed_all_1 or force_gen_inputs_outputs:
        pp_yellow(dot + dot + " force gen new feeds")
        feed_kv = None
    if feed_kv == None:
        feed_kv = gen_feed_kv()
        save_feed_kv(feed_kv)
        feed_kv = load_feed_kv()
    # 预测
    pp_yellow(dot + dot + " checking inference")
    outputs = run_model(feed_kv=feed_kv)
    # pp_tab("fluid output : {}".format(outputs), 1)
    # 重新保存模型
    pp_yellow(dot + dot + " checking model correctness")
    resave_model(feed_kv=feed_kv)
    # 输出加密模型
    encrypt_model()
    # 输出所有中间结果
    pp_yellow(dot + dot + " checking output result of every op")
    save_all_op_output(feed_kv=feed_kv)

    if need_print_mean:
        pp_yellow(dot + dot + " gen check tensor vectors")
        gen_meanname_vectors(mean_dict)

    pp_yellow(dot + dot + " checking fetch info")
    for fetch in fetches:
        fetch_name = fetch.name
        # last_fetch_var_name = fetch_name
        fetch_shape = get_var_shape(fetch_name)
        pp_tab(
            "fetch var name : {}; fetch var shape : {}".format(
                fetch_name, fetch_shape), 1)
    # 输出所有op、var信息
    pp_yellow(dot + dot + " 输出所有op、var信息")

    info_file = open(model_path + "/" + "info.txt", "w")
    for i in range(len(ops)):
        op = ops[i]
        info_file.write("{}th op: type - {}\n".format(i, op.type))
        info_file.write("inputs:\n")
        for var_name in op.input_arg_names:
            try:
                shape = get_var_shape(var_name)
                shape_str = ", ".join(list(map(lambda x: str(x), shape)))
                info_file.write("var {} : {}\n".format(var_name, shape_str))
            except Exception as e:
                print_e(e)
        info_file.write("outputs:\n")
        for var_name in op.output_arg_names:
            try:
                # data = get_var_data(
                #     var_name, feed_kv=feed_kv).flatten().tolist()
                # mean = calc_mean(mean, data)
                shape = get_var_shape(var_name)
                shape_str = ", ".join(list(map(lambda x: str(x), shape)))
                mean = -1.0
                # print(mean_dict)
                if var_name in mean_dict:
                    mean = mean_dict[var_name]
                info_file.write(
                    "var {0:*^20} : {1:-^10}  mean:  {2:5.5f}\n".format(
                        var_name, shape_str, mean))
            except Exception as e:
                print_e(e)

    info_file.close()
    # 开始检查mobile的正确性
    check_mobile()
    # 开始检查lite的正确性
    check_lite_results()


if __name__ == "__main__":
    main()
