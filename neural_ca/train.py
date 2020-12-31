import os
import sys
import logging
import argparse
import contextlib

import wandb
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from neural_ca import util
from neural_ca.models.automata import AutomataModel

### CONSTANTS ###
HEIGHT = 64
WIDTH = 64
BATCH_SIZE = 8
STATE_SIZE = 16
DROP_PROB = 0.5
VAL_STEPS = 250
VIDEO_STEPS = 256
GEN_RANGE = (64, 96)
EMOJI = "🦎"
LR = 2e-3

def make_video(model, image, steps):
    def process_cell(cell):
        rgb_cell = util.to_rgb(cell.numpy()).numpy()
        clipped_cell = np.uint8(rgb_cell.clip(0, 1) * 255)
        unbatched_cell = clipped_cell[0, :, :, :]
        # From channel last to channel first for wandb
        swap_cell = np.moveaxis(unbatched_cell, 2, 0)
        return swap_cell

    cell = util.make_seeds(image.shape, 1, STATE_SIZE)
    video = [process_cell(cell)]
    for i in range(steps - 1):
        cell = model(cell)
        video.append(process_cell(cell))
    return np.stack(video, axis=0)

def log(i, loss, model, image):
    log_data = {
        "step": i,
        "loss": loss,
    }
    if i % VAL_STEPS == 0:
        video = make_video(model, image, VIDEO_STEPS)
        log_data["video"] = wandb.Video(video, format="webm", fps=8)
        tqdm.write(f" - Loss: {loss}, video logged")
    wandb.log(log_data)

def calc_loss(cells, image):
    pixel_delta = util.to_rgba(cells) - image
    loss = tf.reduce_mean(tf.square(pixel_delta))
    return loss

def train(model, optimizer, train_steps, image):
    for i in tqdm(tf.range(train_steps), desc="Training ", leave=True):
        cells = util.make_seeds(image.shape, BATCH_SIZE, STATE_SIZE)
        gen_steps = tf.random.uniform([], GEN_RANGE[0], GEN_RANGE[1], tf.int32)
        with tf.GradientTape() as tape:
            for j in tf.range(gen_steps):
                cells = model(cells)
            loss = calc_loss(cells, image)
        grads = tape.gradient(loss, model.trainable_weights)
        grads = [g/(tf.norm(g) + 1e-8) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        log(i, loss, model, image)

def build_model():
    model = AutomataModel(STATE_SIZE, drop_prob=DROP_PROB)
    return model

def build_optimizer():
    lr_scheduler = tf.optimizers.schedules.PiecewiseConstantDecay(
        [2000], [LR, LR * 0.1]
    )
    optimizer = tf.keras.optimizers.Adam(lr_scheduler)
    return optimizer

def main(args):
    parser = argparse.ArgumentParser('Train Neural CA')
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural-ca")
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None)
    parser.add_argument(
        "--train_steps",
        type=int,
        default=2000)
    args = parser.parse_args(args)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name
    )

    model = build_model()
    optimizer = build_optimizer()
    image = util.load_emoji(EMOJI)
    train(model, optimizer, args.train_steps, image)

if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    main(sys.argv[1:])

