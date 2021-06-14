import requests
from PIL import Image
from io import BytesIO

import numpy as np
import tensorflow as tf

def load_image_from_url(url):
    """ Loads an image from url """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def process_image(img, size=64):
    """ Processes image for training """
    img.thumbnail((size, size), Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    # Scale RGB by alpha
    # From original implementation, unclear exactly why
    img[..., :3] *= img[..., 3:]
    return img

def load_emoji(emoji):
    """ Loads an emoji """

    # Black magic from original implemenetation
    code = hex(ord(emoji))[2:].lower()

    url = ('https://raw.githubusercontent.com/googlefonts/noto-emoji/'
           f'948b1a7f1ed4ec7e27930ad8e027a740db3fe25e/png/128/emoji_u{code}.png')
    emoji = load_image_from_url(url)
    processed_emoji = process_image(emoji)
    return processed_emoji

def to_alpha(image):
    return tf.clip_by_value(image[..., 3:4], 0.0, 1.0)

def to_rgba(image):
    return image[:, :, :, :4]

def to_rgb(image):
    rgb = image[:, :, :, :3]
    alpha = to_alpha(image)
    return 1.0 - alpha + rgb
