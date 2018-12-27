# coding=utf-8
import os

from core import framework_pb2 as framework_pb2

MEM16_0___ = 'MEM16_0'
MEM16_1___ = 'MEM16_1'
MEM1_0___ = 'MEM1_OUT'
MEM1_1___ = MEM1_0___
MEM1_2___ = 'MEM1_FEED'

last_mem = MEM16_0___


class Optimiser:
    next_input = MEM16_1___
    next_output = MEM16_0___

    def __init__(self):
        pass

    def inner_pack_16_argu(self, argu, input, value):
        if 'tmp_0' in argu:
            del input.arguments[:]
            input.arguments.extend([value])
            return True

        if 'tmp_1' in argu:
            del input.arguments[:]
            input.arguments.extend([value])
            return True
        return False

    def replace_argu(self, argu, input, is_conv, is_input):
        if 'Convolution10.conv2d.output.1.tmp_0' in argu:
            del input.arguments[:]
            input.arguments.extend([MEM1_0___])
            return
        if 'Convolution10.conv2d.output.1.tmp_1' in argu:
            del input.arguments[:]
            input.arguments.extend([MEM1_1___])
            return
        if 'Eltwise1.add.output.1.tmp_0' in argu:
            del input.arguments[:]
            input.arguments.extend([MEM1_0___])
            return

        if 'Input1' in argu:
            del input.arguments[:]
            input.arguments.extend([MEM1_2___])
            return

        if is_conv:
            # 卷积
            if is_input:
                # 输入
                if self.inner_pack_16_argu(argu, input, self.next_input):
                    if self.next_input == MEM16_0___:
                        self.next_input = MEM16_1___
                    else:
                        self.next_input = MEM16_0___
                    return
                pass
            else:
                # 输出
                if self.inner_pack_16_argu(argu, input, self.next_input):
                    return
                pass

        else:
            # 输出
            if self.inner_pack_16_argu(argu, input, self.next_input):
                return
            pass

    def replace_var(self, blocks, name):
        for block in blocks:
            for var in block.vars:
                if var.name == name:

                    if 'Convolution10.conv2d.output.1.tmp_0' in var.name:
                        var.name = MEM1_0___
                        break
                    if 'Convolution10.conv2d.output.1.tmp_1' in var.name:
                        var.name = MEM1_1___
                        break
                    if 'Eltwise1.add.output.1.tmp_0' in var.name:
                        var.name = MEM1_0___
                        break

                    if 'Input1' in var.name:
                        var.name = MEM1_2___
                        break

                    if 'tmp_0' in var.name:
                        var.name = MEM16_0___
                        break

                    if 'tmp_1' in var.name:
                        var.name = MEM16_1___
                        break

    def read_model(self, model_path):
        print('read_model.')
        path_8 = unicode(model_path, 'utf8')

        try:
            with open(path_8, "rb") as f_model:
                print get_file_size(model_path)
                desc = framework_pb2.ProgramDesc()
                desc.ParseFromString(f_model.read())

                blocks = desc.blocks
                for block in blocks:
                    # print block
                    ops = block.ops
                    for op in ops:
                        print '----------op: ---------------'
                        print op.type
                        # print op.inputs
                        # print op.outputs

                        # print op
                        if 1:
                            print 'input ----:::'
                            for input in op.inputs:
                                # print 'argument:'
                                # print 'an input :'
                                for argu in input.arguments:
                                    if 'weights' in argu:
                                        continue
                                    if 'biases' in argu:
                                        continue
                                    self.replace_var(blocks, argu)
                                    # if argu.startswith('Convolution'):
                                    self.replace_argu(argu, input, op.type == 'conv2d', 1)

                            print 'output ----:::'

                            for output in op.outputs:
                                # print 'argument:'
                                # print 'an input :'
                                for argu in output.arguments:
                                    if 'weights' in argu:
                                        continue
                                    if 'biases' in argu:
                                        continue
                                    self.replace_var(blocks, argu)
                                    self.replace_argu(argu, output, op.type == 'conv2d', 0)

                                    # print input
                            # print input.arguments

                print desc

                f = open(
                    "/Users/xiebaiyuan/PaddleProject/GITHUB/PaddleMobileModelTools/python/modeltools/tools/optimised/model",
                    "wb")
                f.write(desc.SerializeToString())
                f.close()

        except IOError:
            print ": File not found."


def get_file_size(file_path):
    file_path = unicode(file_path, 'utf8')
    fsize = os.path.getsize(file_path)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)


path = '/Users/xiebaiyuan/PaddleProject/paddle-mobile/test/models/superresoltion_backuo/model'
Optimiser().read_model(path)
