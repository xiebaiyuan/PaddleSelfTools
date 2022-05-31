# %% [markdown]
# ### Create Dir IF NOT EXISTS

# %%
def create_dir_if_not_exist(dir_path):
    import os
    if dir_path != "" and not os.path.exists(dir_path):
        os.makedirs(dir_path)


# %%
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
# def print_e(e):
#     if not check_exception:
#         return
#     print('str(Exception):\t', str(Exception))
#     print('str(e):\t\t', str(e))
#     print('repr(e):\t', repr(e))


# %%
def pp_tab(x, level=2):
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


# %% [markdown]
# ### GLOBAL INFOS

# %%
# model_path=model_type+"/opted/"
# model_path = "./ar_cheji_day/"
# model_path = "./model_cheji/"
model_path = "./cheji_ar_0517/"
# model_path = "./ar_cheji_day/ar_cheji_day_test_model/"
model_filename = "model.pdmodel"
params_filename = "model.pdiparams"

# %%
import paddle.fluid as fluid
import paddle

paddle.enable_static()


def load_inference_model(model_path, model_filename, params_filename):
    exe = fluid.Executor(fluid.CPUPlace())
    net_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(dirname=model_path, executor=exe,
                                                                                  model_filename=model_filename,
                                                                                  params_filename=params_filename)
    return exe, net_program, feed_target_names, fetch_targets


exe, net_program, feed_target_names, fetch_targets = load_inference_model(model_path, model_filename, params_filename)


# %% [markdown]
# ### GET VAR SHAPES

# %%
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


# %% [markdown]
# ### MODEL INFO PARSE

# %%
# print("feed_target_names========>")
# print(feed_target_names)

# print("fetch_targets========>")
# print(fetch_targets)

# print("prog========>")
# print(net_program)
for name in feed_target_names:
    print(name)
    shape = get_var_shape(name)
    print(get_var_shape_str(shape))


def get_feed_shape_str():
    shapes_str = ""
    for name in feed_target_names:
        shape_str = get_var_shape_str(get_var_shape(name))
    shapes_str += shape_str;
    shapes_str += ":";
    return shapes_str


feed_shape_str = get_feed_shape_str()
print(feed_shape_str)

origin_fetch_names = []
for fetch_target in fetch_targets:
    # print(name)
    origin_fetch_names.append(fetch_target.name.replace("/", "_"))

print("origin_fetch_names========>" + str(origin_fetch_names))

# %% [markdown]
# ### COMMON TOOLS

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

GLB_model_path = ''
GLB_arg_name = ''
GLB_batch_size = 1


def load_inference_model(model_path, exe):
    '''
    '''
    model_abs_path = os.path.join(model_path, 'ar_cheji_day/model')
    param_abs_path = os.path.join(model_path, 'ar_cheji_day/params')
    if os.path.exists(model_abs_path) and os.path.exists(param_abs_path):
        return fluid.io.load_inference_model(model_path, exe, 'model', 'params')
    else:
        return fluid.io.load_inference_model(model_path, exe)


def feed_ones(block, feed_target_names, batch_size=1):
    """ 
    """
    feed_dict = dict()

    def set_batch_size(shape, batch_size):
        if shape[0] == -1:
            shape[0] = batch_size
        return shape

    def fill_ones(var_name, batch_size):
        var = block.var(var_name)
        np_shape = set_batch_size(list(var.shape), 1)
        var_np = {
            core.VarDesc.VarType.BOOL: np.bool_,
            core.VarDesc.VarType.INT32: np.int32,
            core.VarDesc.VarType.INT64: np.int64,
            core.VarDesc.VarType.FP16: np.float16,
            core.VarDesc.VarType.FP32: np.float32,
            core.VarDesc.VarType.FP64: np.float64,
        }
        np_dtype = var_np[var.dtype]
        return np.ones(np_shape, dtype=np_dtype)

    for feed_target_name in feed_target_names:
        feed_dict[feed_target_name] = fill_ones(feed_target_name, batch_size)

    return feed_dict


def feed_randn(block, feed_target_names, batch_size=1, need_save=True):
    """ 
    """
    feed_dict = dict()

    def set_batch_size(shape, batch_size):
        if shape[0] == -1:
            shape[0] = batch_size
        return shape

    def fill_randn(var_name, batch_size, need_save):
        var = block.var(var_name)
        np_shape = set_batch_size(list(var.shape), 1)
        var_np = {
            core.VarDesc.VarType.BOOL: np.bool_,
            core.VarDesc.VarType.INT32: np.int32,
            core.VarDesc.VarType.INT64: np.int64,
            core.VarDesc.VarType.FP16: np.float16,
            core.VarDesc.VarType.FP32: np.float32,
            core.VarDesc.VarType.FP64: np.float64,
        }
        np_dtype = var_np[var.dtype]
        numpy_array = np.random.random(np_shape).astype(np.float32)
        if need_save is True:
            numpy_to_txt(numpy_array, 'feed_' + var_name + '.txt', True, model_path)
        return numpy_array

    for feed_target_name in feed_target_names:
        feed_dict[feed_target_name] = fill_randn(feed_target_name, batch_size, need_save)
    return feed_dict


def draw(block, filename='debug'):
    """
    """
    dot_path = './' + filename + '.dot'
    pdf_path = './' + filename + '.pdf'
    debugger.draw_block_graphviz(block, path=dot_path)
    cmd = ["dot", "-Tpdf", dot_path, "-o", pdf_path]
    subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


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


def numpy_var(scope, var_name):
    """
    get numpy data by the name of var.
    """
    if hasattr(fluid.executor, '_fetch_var'):
        numpy_array = fluid.executor._fetch_var(var_name, scope, True)
    elif hasattr(fluid.executor, 'fetch_var'):
        numpy_array = fluid.executor.fetch_var(var_name, scope, True)
    else:
        raise NameError('ERROR: Unknown Fluid version.')
    return numpy_array


def var_dtype(block, var_name):
    """
    get dtype of fluid var.
    """
    var = block.var(var_name)
    return var.dtype


def print_ops_type(block):
    """
    """

    def ops_type(block):
        ops = list(block.ops)
        cache = []
        for op in ops:
            if op.type not in cache:
                cache.append(op.type)
        return cache

    type_cache = ops_type(block)
    for op_type in type_cache:
        print(op_type)


all_true_var_resuls = {}


def print_results(results, fetch_targets, need_save=False):
    """
    """
    for result in results:
        idx = results.index(result)
        # print(fetch_targets[idx])
        A = np.array(result)
        # print(results[idx])
        all_true_var_resuls[fetch_targets[idx].name.replace('/', '_')] = A
        print("==========={}============= std and mean ========>".format(fetch_targets[idx].name))
        print(fetch_targets[idx].name)

        print('mean={}'.format(A.flatten().mean()))
        print('std={}'.format(np.std(A)))
        if need_save is True:
            numpy_to_txt(result, fetch_targets[idx].name.replace('/', '_'), True, model_path)


def numpy_to_txt(numpy_array, save_name, print_shape=True, save_dir=""):
    """
    transform numpy to txt.
    """
    target_path = save_dir + "/" + "vars/"

    create_dir_if_not_exist(target_path)

    np_array = np.array(numpy_array)
    fluid_fetch_list = list(np_array.flatten())
    fetch_txt_fp = open(target_path + "/" + save_name + '.txt', 'w')
    for num in fluid_fetch_list:
        fetch_txt_fp.write(str(num) + '\n')
    if print_shape is True:
        fetch_txt_fp.write('Shape: (')
        for val in np_array.shape:
            fetch_txt_fp.write(str(val) + ', ')
        fetch_txt_fp.write(')\n')
    fetch_txt_fp.close()


def fluid_inference_test(model_path):
    """
    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        [net_program,
         feed_target_names,
         fetch_targets] = load_inference_model(model_path, exe)
        print(net_program)
        global_block = net_program.global_block()
        draw(net_program.block(0))
        feed_list = feed_ones(global_block, feed_target_names, 1)
        # feed_list = feed_randn(global_block, feed_target_names, 1, need_save=True)
        fetch_targets = fetch_tmp_vars(global_block, fetch_targets, [GLB_arg_name])
        results = exe.run(program=net_program,
                          feed=feed_list,
                          fetch_list=fetch_targets,
                          return_numpy=False)
        # for var_ in net_program.list_vars():
        #  print var_
        # print list(filter(None, net_program.list_vars()))
        fluid.io.save_params(executor=exe, dirname="./123", main_program=net_program)
        print_results(results, fetch_targets, need_save=True)


# if __name__ == "__main__":
#    arg_parser = argparse.ArgumentParser('fluid feed_ones')

#    arg_parser.add_argument('--model_path', type=str, required=True)
#    arg_parser.add_argument('--arg_name', type=str)
#    arg_parser.add_argument('--batch_size', type=int)

#    args = arg_parser.parse_args()
#    paddle.enable_static()

#    GLB_model_path = args.model_path
#    if args.arg_name is not None:
#        GLB_arg_name = args.arg_name
#    if args.batch_size is not None:
#        GLB_batch_size = args.batch_size

#    fluid_inference_test(GLB_model_path)


# %%
# draw(prog.block(0),"draw_text")
print(net_program.block(0).ops[1].desc.outputs()['Output'])

ops = net_program.current_block().ops
vars = net_program.current_block().vars

addition_tmp_vars = []
for op in ops:
    for var_name in op.output_arg_names:
        if var_name == "fetch":
            continue
        var = vars[var_name]
        if not var.persistable:
            # print("has found non-persistable output var : {}".format(
            #     var_name))
            # put non-presistable output var into tmp list
            addition_tmp_vars.append(var_name)
            print("shape of {} is {}".format(var_name, shape))
            # var.persistable = False

print(addition_tmp_vars)

# %% [markdown]
# ### GENERATE TEST MODELS

# %%
global_block = net_program.global_block()
fetch_targets = fetch_tmp_vars(global_block, fetch_targets, addition_tmp_vars)

feed_list = feed_ones(global_block, feed_target_names, 1)

results = exe.run(program=net_program,
                  feed=feed_list,
                  fetch_list=fetch_targets,
                  return_numpy=False)
# Print Results
print_results(results, fetch_targets, need_save=False)

test_model_path = model_path + "/" + model_path + "_test_model"
create_dir_if_not_exist(test_model_path)
fluid.io.save_inference_model(dirname=test_model_path,
                              feeded_var_names=feed_target_names,
                              target_vars=fetch_targets,
                              executor=exe,
                              main_program=net_program,
                              model_filename=model_filename,
                              params_filename=params_filename)

# %%

# %%
import os


def file_name_walk(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print("root", root)  # 当前目录路径
        print("dirs", dirs)  # 当前路径下所有子目录
        print("files", files)  # 当前路径下所有非目录子文件


def file_name_listdir(file_dir):
    result_files = []
    for files in os.listdir(file_dir):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
        # print("files", files)
        result_files.append(files)
    return result_files
