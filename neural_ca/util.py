import numpy as np
import tensorflow as tf

def load_image_from_url(url, size=64):
    """ Loads an image from url """
    pass

def load_emoji(emoji):
    """ Loads an emoji """
    pass

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
