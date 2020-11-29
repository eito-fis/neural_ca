import requests
from PIL import Image
from io import BytesIO

import numpy as np
import tensorflow as tf

def load_image_from_url(url):
    """ Loads an image from url """
    response = requests.get(url)
    img = Image.open(response.raw)
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
    url = 'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u{code}.png'
    emoji = load_image_from_url(url)
    processed_emoji = process_image(emoji)
    return processed_emoji

def make_seeds(shape, batch_size, state_size):
    """ Makes batch of seeds """
    assert len(shape == 3)
    height, width = shape[:2]
    seeds = np.zeros([batch_size, height, width, state_size])
    seeds[:, HEIGHT // 2, WIDTH // 2, 3:] += 1
    seeds = tf.convert_to_tensor(seeds)
    return seeds

def to_rgb(image):
    return image[:, :, :, :3]

def to_rgba(image):
    return image[:, :, :, :4]
