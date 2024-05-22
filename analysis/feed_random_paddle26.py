#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
A separate Fluid test file for feeding specific data.
'''

import argparse
import sys
import os
import paddle
import paddle.framework

import numpy as np
from scipy.stats import skew, kurtosis
from paddle.static import InputSpec

GLB_model_path = ''
GLB_arg_name = ''
GLB_batch_size = 1


def load_inference_model(model_path, exe):
    """
    Load the inference model and execute it.
    """
    return paddle.static.load_inference_model(model_path, exe)


#
def compute_features(data, des=""):
    """
    Compute features for a batch of data.
    """
    features = {}
    features['mean'] = np.mean(data)
    features['std_dev'] = np.std(data)
    features['skewness'] = skew(data)
    features['kurtosis'] = kurtosis(data)
    features['min'] = np.min(data)
    features['max'] = np.max(data)
    features['range'] = np.max(data) - np.min(data)
    features['25th_percentile'] = np.percentile(data, 25)
    features['median'] = np.percentile(data, 50)
    features['75th_percentile'] = np.percentile(data, 75)
    features['iqr'] = np.percentile(data, 75) - np.percentile(data, 25)  # Interquartile range

    print("\n" + des)
    print("mean: {:.4f}".format(features['mean']))
    print("std_dev: {:.4f}".format(features['std_dev']))
    print("skewness: {:.4f}".format(features['skewness']))
    print("kurtosis: {:.4f}".format(features['kurtosis']))
    print("min: {:.4f}".format(features['min']))
    print("max: {:.4f}".format(features['max']))
    print("range: {:.4f}".format(features['range']))
    print("25th_percentile: {:.4f}".format(features['25th_percentile']))
    print("median: {:.4f}".format(features['median']))
    print("75th_percentile: {:.4f}".format(features['75th_percentile']))
    print("iqr: {:.4f}".format(features['iqr']))
    return features


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
        return np.ones(np_shape, dtype=var.dtype)

    for feed_target_name in feed_target_names:
        feed_dict[feed_target_name] = fill_ones(feed_target_name, batch_size)

    return feed_dict


def feed_randn(block, feed_target_names, batch_size=1, need_save=True):
    """
    Fill the network input with randomly generated data, converting integers to floats where applicable.
    """
    feed_dict = dict()

    def set_batch_size(shape, batch_size):
        if shape[0] == -1:
            shape[0] = batch_size
        if shape[1] == -1:
            shape[1] = batch_size
        return shape

    def fill_randn(var_name, batch_size):
        var = block.var(var_name)
        print("update var shape: {} of type {}".format(var_name, type(var.dtype)))
        np_shape = set_batch_size(list(var.shape), batch_size)
        print(np_shape)

        var_np = {
            paddle.bool: np.bool_,
            paddle.int32: np.int32,
            paddle.int64: np.int64,
            paddle.float16: np.float16,
            paddle.float32: np.float32,
            paddle.float64: np.float64,
        }
        np_dtype = var_np[var.dtype]

        # 设置随机数种子
        np.random.seed(42)

        # 对浮点类型进行整数到浮点的转换
        if np.issubdtype(np_dtype, np.floating):
            max_int = 9999
            print(np_shape)
            random_ints = np.random.randint(0, max_int + 1, np_shape)
            random_floats = random_ints.astype(np_dtype) / max_int
            compute_features(random_floats.flatten(), "生成随机数特征:")
            return random_floats
        else:
            # 非浮点类型直接生成随机整数
            return np.random.randint(0, 100, np_shape, dtype=np_dtype)

    for feed_target_name in feed_target_names:
        feed_dict[feed_target_name] = fill_randn(feed_target_name, batch_size)

    return feed_dict


def draw(block, filename='debug'):
    """
    """
    # dot_path = './' + filename + '.dot'
    # pdf_path = './' + filename + '.pdf'
    # debugger.draw_block_graphviz(block, path=dot_path)
    # cmd = ["dot", "-Tpdf", dot_path, "-o", pdf_path]
    # subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


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


# def var_dtype(block, var_name):
#     """
#     get dtype of fluid var.
#     """
#     var = block.var(var_name)
#     return var.dtype
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


def calculate_statistics(data):
    """
    计算并返回给定数据的均值、标准差、偏度和峰度。
    忽略数据中的 NaN 值。

    参数:
    data (array-like): 输入的数字数据，可以是列表或 numpy 数组。

    返回:
    dict: 包含均值、标准差、偏度和峰度的字典。
    """
    # 检查Nan并且打印出位置
    print(np.argwhere(np.isnan(data)))

    # 将输入数据转换为 numpy 数组，确保处理的一致性
    data_array = np.array(data)

    # 计算均值和标准差，忽略 NaN
    mean_val = np.nanmean(data_array)
    std_val = np.nanstd(data_array)

    # 清理数据：将 NaN 替换为均值
    clean_data = np.where(np.isnan(data_array), mean_val, data_array)

    # 计算偏度和峰度，忽略 NaN
    skew_val = skew(clean_data, nan_policy='omit')
    kurtosis_val = kurtosis(clean_data, nan_policy='omit')

    # 创建包含统计结果的字典
    statistics = {
        'mean': mean_val,
        'std': std_val,
        'skew': skew_val,
        'kurtosis': kurtosis_val
    }

    return statistics


# 示例用法
# result = [1, 2, np.nan, 4, 5]
# stats = calculate_statistics(result)
# print("统计结果:", stats)
def print_results(results, fetch_targets, need_save=False):
    """
    """
    for result in results:
        idx = results.index(result)
        print(fetch_targets[idx])
        A = np.array(result)
        # statistics = calculate_statistics(A.flatten())
        # print(statistics)

        compute_features(A.flatten(), "输出 " + str(idx) + " (" + fetch_targets[idx].name + ") 特征: ")
        if need_save is True:
            numpy_to_txt(result, 'result_' + fetch_targets[idx].name.replace('/', '_'), True)


def numpy_to_txt(numpy_array, save_name, print_shape=True):
    """
    transform numpy to txt.
    """
    np_array = np.array(numpy_array)
    fluid_fetch_list = list(np_array.flatten())
    fetch_txt_fp = open(save_name + '.txt', 'w')
    for num in fluid_fetch_list:
        fetch_txt_fp.write(str(num) + '\n')
    if print_shape is True:
        fetch_txt_fp.write('Shape: (')
        for val in np_array.shape:
            fetch_txt_fp.write(str(val) + ', ')
        fetch_txt_fp.write(')\n')
    fetch_txt_fp.close()


def inference_test(model_path):
    """
    """
    exe = paddle.static.Executor(paddle.CPUPlace())

    [net_program, feed_target_names, fetch_targets] = load_inference_model(model_path, exe)

    print(net_program)
    global_block = net_program.global_block()
    draw(net_program.block(0))
    feed_list = feed_randn(global_block, feed_target_names, GLB_batch_size, need_save=True)
    fetch_targets = fetch_tmp_vars(global_block, fetch_targets, [GLB_arg_name])
    results = exe.run(program=net_program,
                      feed=feed_list,
                      fetch_list=fetch_targets,
                      return_numpy=False)
    # for var_ in net_program.list_vars():
    #  print var_
    # print list(filter(None, net_program.list_vars()))
    # paddle.static.save_inference_model(path_prefix="./debug_model", feed_vars=feed_target_names,
    #                                    fetch_vars=fetch_targets,
    #                                    executor=exe)

    print_results(results, fetch_targets, need_save=True)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser('paddle2.6 feed_random_paddle26')

    arg_parser.add_argument('--model_path', type=str, required=True)
    arg_parser.add_argument('--arg_name', type=str)
    arg_parser.add_argument('--batch_size', type=int)

    args = arg_parser.parse_args()
    paddle.enable_static()

    GLB_model_path = args.model_path
    if args.arg_name is not None:
        GLB_arg_name = args.arg_name
    if args.batch_size is not None:
        GLB_batch_size = args.batch_size

    inference_test(GLB_model_path)
