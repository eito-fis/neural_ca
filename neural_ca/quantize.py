import os
import sys
import argparse

import wandb
from tqdm import tqdm
import tensorflow as tf
import moviepy.editor as mpy

from neural_ca import util
from neural_ca.pools import SamplePool, VideoPool
from neural_ca.models.automata import AutomataModel

HEIGHT = 64
WIDTH = 64
BATCH_SIZE = 8
STATE_SIZE = 16
DROP_PROB = 0.5
VAL_STEPS = 250
VIDEO_STEPS = 736
GEN_RANGE = (64, 96)
EMOJI = "🦎"
LR = 2e-3
POOL_SIZE = 1024

def save_video(model):
    pool = SamplePool(POOL_SIZE, STATE_SIZE, EMOJI)
    video = util.video.make_video(model, pool, VIDEO_STEPS)
    clip = mpy.ImageSequenceClip(video, fps=16)
    filename = os.path.join("logging", "quantize", "test.mp4")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    clip.write_videofile(filename, logger=None)


def main(args):
    parser = argparse.ArgumentParser('Quantize Neural CA')
    parser.add_argument(
        "--load",
        type=str,
        default=None)
    args = parser.parse_args(args)
    assert args.load

    model = tf.keras.models.load_model(args.load)
    save_video(model)


if __name__ == '__main__':
    main(sys.argv[1:])
