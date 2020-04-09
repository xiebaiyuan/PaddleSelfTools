# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import onnx
from onnx import shape_inference
import warnings
from onnx_tf.backend import prepare
import numpy as np


# %%
def stride_print(input):
    tensor = input.flatten().tolist()
    length = len(tensor)
    size = 20
    stride = length//size
    if stride == 0:
        stride = 1
    size = length // stride
    nums = []
    for i in range(0, size):
        item = tensor[i * stride]
        # nums.append(str(i * stride) + ": " + str(item))
        nums.append(str(item))
    print(nums)
    # for i in range(0, size):
    #     item = tensor[i * stride]
    #     print ("{} ".format(item),end="")


# %%
diff_threadhold = 0.05


def compare(input):
    stride_print(input)
    tensor = input.flatten().tolist()
    length = len(tensor)
    size = 20
    stride = length//size
    if stride == 0:
        stride = 1
    size = length // stride
    nums = []
    for i in range(0, size):
        item = tensor[i * stride]
        # nums.append(str(i * stride) + ": " + str(item))
        nums.append(item)

    for i in range(0, size):
        right_v = nums[i]
        paddle_v = float(input_paddle[i])
        if (abs(right_v-paddle_v) > diff_threadhold):
            print("err at {} {} {} ".format(i, right_v, paddle_v))


# %%
model = onnx.load("v18_7_6_2_leakyReLU_rgb_mask_test_t2.onnx")
onnx.checker.check_model(model)
inferred_model = shape_inference.infer_shapes(model)


# %%
model.graph.output.extend(inferred_model.graph.value_info)


# %%
warnings.filterwarnings('ignore')
tfm = prepare(model)
# input = np.fromfile('input', dtype=np.float32).reshape(1, 3, 256, 256)

input = np.loadtxt('./input_1_3_256_256',
                   dtype=np.float32).reshape(1, 3, 256, 256)
res = tfm.run(input)


# %%
input_paddle = "0.53125 0.549316 0.558594 0.677246 0.470703 0.634766 0.540039 0.566406 0.495605 0.597168 0.602539 0.480957 0.448486 0.553711 0.474365 0.612793 0.609863 0.518555 0.617188 0.505371 0.504395".split(
    " ")
compare(res["mask"])


# %%
input_paddle = "0.245117 -0.222656 0.0887451 0.803711 0.639648 0.0995483 0.807129 -0.224609 -0.267578 0.33667 0.372559 -0.353516 0.343262 0.549805 0.344971 0.503906 0.152466 -0.0531616 0.0315247 -0.0397034 -0.218262".split(
    " ")
compare(res["rgb"])
