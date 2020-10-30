# -*- coding: utf-8 -*
# %%
import os
import sys
import math
import subprocess
import numpy as np
import paddle.fluid as fluid
from time import sleep

IS_DEBUG = False
is_sample_step = True
sample_step = 1
sample_num = 20

need_save = False
diff_threshold = 0.1
feed_all_1 = True
force_gen_inputs_outputs = False
need_print_mean = True
show_correct_check = False
need_check_mobile = False

need_wanted = False
wanted_list = [
    "blocks.2.0.se.conv_reduce.tmp_2", "blocks.2.0.se.conv_reduce.tmp_1",
    "blocks.2.0.se.conv_reduce.tmp_0"
]

model_path = "/data/paddle_face_model/saved-20201028-185037"

checked_model_path = model_path + "/" + "checked_model"
feed_path = model_path + "/" + "feeds"
output_path = model_path + "/" + "outputs"

mobile_exec_root = "/data/local/tmp/bin"

###### mobile config ######
# TODO 这个lod再说吧...
is_lod = False
fast_check = False

need_encrypt = False
check_exception = False
checked_encrypt_model_path = "checked_encrypt_model"
output_var_filter = []
output_key_filter = {}

check_shape = False

# architecture = "arm-v7a"
# # architecture = "arm-v8a"
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
    outputs = ""
    while True:
        line = pipe.stdout.readline()
        if not line:
            break
        # print(line.decode("utf-8"))
        outputs += line.decode("utf-8")
    return outputs
    # return pipe.stdout.read().decode("utf-8")


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
            "{0:30}  {1:30.5f}   {2:30}".format(name, mean,
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

        if fast_check:
            if var_name not in fetch_names and var_name not in feed_names:
                continue
        try:
            data_np_arr = get_var_data(var_name, feed_kv=feed_kv)
            data = data_np_arr.flatten().tolist()

            # 计算均值,方差
            if need_print_mean:
                np_mean = np.mean(data_np_arr)
                np_var = np.var(data_np_arr)

                #   template <typename T>
                #   double compute_average_grow_rate(const T* in, const size_t length) {
                #     const double eps = 1e-5;
                #     double ave_grow_rate = 0.0f;
                #     for (size_t i = 1; i < length; ++i) {
                #       ave_grow_rate += (in[i] - in[i - 1]) / (in[i - 1] + eps);
                #     }
                #     ave_grow_rate /= length;
                #     return ave_grow_rate;
                #   }
                # wt = np.arange(data_np_arr.size)
                eps = 1e-5
                ave_grow_rate = 0.0
                for index in range(1, len(data)):
                    ave_grow_rate += (data[index] - data[index - 1]) / (
                        data[index - 1] + eps)

                # print(wt)
                # avg = np.average(data_np_arr, weights=wt)
                pp_green(
                    "{0:<30} {1:<25.5f}{2:<25.5f} {3:<25.5f} {4:30}".format(
                        var_name, np_mean, np_var, ave_grow_rate,
                        str(get_var_shape(var_name))), 1)

            sample = tensor_sample(data)
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

    pp_yellow(dot + dot + " checking fetch info")
    for fetch in fetches:
        fetch_name = fetch.name
        # last_fetch_var_name = fetch_name
        fetch_shape = get_var_shape(fetch_name)
        pp_tab(
            "fetch var name : {}; fetch var shape : {}".format(
                fetch_name, fetch_shape), 1)
    # 输出所有op、var信息到文件
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


if __name__ == "__main__":
    main()
