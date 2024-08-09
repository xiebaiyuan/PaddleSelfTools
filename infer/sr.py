import paddle
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global configuration
GLB_model_path = 'path_to_your_model'
GLB_batch_size = 1


def load_and_preprocess_image(image_path):
    """
    Load an image, resize it, and preprocess it for model inference.
    """
    # Read image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at the specified path.")

    # Convert color from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the required size (e.g., 224x224)
    image = cv2.resize(image, (224, 224))

    # Normalize the image by dividing by 255 to scale pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Convert image to CHW format
    image = np.transpose(image, (2, 0, 1))

    # Add batch dimension
    image = image[np.newaxis, :]

    return image


def load_inference_model(model_path):
    """
    Load the inference model.
    """
    # Load model using PaddlePaddle
    exe = paddle.static.Executor(paddle.CPUPlace())
    [program, feed_target_names, fetch_targets] = paddle.static.load_inference_model(model_path, exe)
    return exe, program, feed_target_names, fetch_targets


def infer_image(image, exe, program, feed_target_names, fetch_targets):
    """
    Perform inference on the image.
    """
    result = exe.run(program,
                     feed={feed_target_names[0]: image},
                     fetch_list=fetch_targets)
    return result


def display_image(image):
    """
    Display the image using matplotlib.
    """
    # Squeeze batch dimension and transpose from CHW to HWC for displaying
    image = np.squeeze(image).transpose((1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def main(image_path):
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)

    # Load the model
    exe, program, feed_target_names, fetch_targets = load_inference_model(GLB_model_path)

    # Perform inference
    output = infer_image(image, exe, program, feed_target_names, fetch_targets)

    # Display output (assuming output is an image tensor)
    display_image(output)


if __name__ == "__main__":
    image_path = 'path_to_your_image.jpg'
    main(image_path)