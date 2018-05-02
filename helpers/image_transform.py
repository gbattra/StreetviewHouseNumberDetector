# image preprocessor object for cleaning input data
import numpy as np
from skimage.transform import resize as ski_resize


def square_crop_image(image):
    image_shape = image.shape
    [long_side, short_side] = [image_shape[0], image_shape[1]] if image_shape[0] > image_shape[1] else [image_shape[1], image_shape[0]]
    indent_size = int((long_side - short_side) / 2)
    cropped_image = image[indent_size:(long_side - indent_size), :, :] if long_side == image_shape[0] else image[:, indent_size:(long_side - indent_size), :]

    return cropped_image


def zero_pad(inputs, pad_size, is_batch=True):
    if is_batch:
        padded_image = np.pad(inputs, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant', constant_values=0)
    else:
        padded_image = np.pad(inputs, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant', constant_values=0)

    return padded_image


def resize_image(image, new_size):
    resized_image = ski_resize(image, (new_size[0], new_size[1]))

    return resized_image
