#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'xiebaiyuan'

import framework_pb2
import os


def read_model(model_path):
    print('read_model.')
    try:
        with open(model_path, "rb") as f_model:
            print(get_file_size(model_path))
            desc = framework_pb2.ProgramDesc()
            desc.ParseFromString(f_model.read())
            print(desc)
            # print desc.blocks

    except IOError:
        print(": File not found.")


def get_file_size(file_path):
    fsize = os.path.getsize(file_path)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)


if __name__ == '__main__':
    path = 'Your model path .'
    read_model(path)
