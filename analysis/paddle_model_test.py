import paddle
import numpy as np

path_prefix = "/Users/baidu/Downloads/iosyolov7/model"
paddle.enable_static()
exe = paddle.static.Executor(paddle.CPUPlace())

[inference_program, feed_target_names, fetch_targets] = (
    paddle.static.load_inference_model(path_prefix, exe))

tensor_img = np.array(np.random.random((1, 3, 384, 384)), dtype=np.float32)
results = exe.run(inference_program,
                  feed={feed_target_names[0]: tensor_img},
                  fetch_list=fetch_targets)
