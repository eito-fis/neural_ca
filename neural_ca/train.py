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

### CONSTANTS ###
HEIGHT = 64
WIDTH = 64
BATCH_SIZE = 8
STATE_SIZE = 16
DROP_PROB = 0.5
VAL_STEPS = 250
VIDEO_STEPS = 256
GEN_RANGE = (1, 4)
EMOJI = "🦎"
LR = 2e-3
POOL_SIZE = 1024

FRAME_STRIDE = 1
SKIP_RANGE = (1, 16)
VIDEO = "data/waves.mp4"

def log(i, loss, model, pool):
    log_data = {
        "step": i,
        "loss": loss,
    }
    if i % VAL_STEPS == 0:
        video = util.video.make_video(model, pool, VIDEO_STEPS)
        clip = mpy.ImageSequenceClip(video, fps=16)
        filename = os.path.join("logging", wandb.run.name, str(i.numpy()) + ".mp4")
        clip.write_videofile(filename, logger=None)
        log_data["video"] = wandb.Video(filename)
        tqdm.write(f" - Loss: {loss}, {filename} logged")
    wandb.log(log_data)

def calc_loss(cells, target):
    pixel_delta = util.image.to_rgba(cells) - target
    loss = tf.reduce_mean(tf.square(pixel_delta))
    return loss

def train(model, optimizer, train_steps, pool):
    for i in tqdm(tf.range(train_steps), desc="Training ", leave=True):
        cells, target, idxs = pool.sample(BATCH_SIZE)
        gen_steps = tf.random.uniform([], GEN_RANGE[0], GEN_RANGE[1], tf.int32)
        with tf.GradientTape() as tape:
            for _ in tf.range(gen_steps):
                cells = model(cells)
            loss = calc_loss(cells, target)
        grads = tape.gradient(loss, model.trainable_weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        pool.update(cells, idxs)
        log(i, loss, model, pool)

def build_model():
    model = AutomataModel(STATE_SIZE, drop_prob=DROP_PROB)
    return model

def build_optimizer():
    lr_scheduler = tf.optimizers.schedules.PiecewiseConstantDecay(
        [2000], [LR, LR * 0.1]
    )
    optimizer = tf.keras.optimizers.Adam(lr_scheduler)
    return optimizer

def build_pool(pool_type="EMOJI"):
    if pool_type == "EMOJI":
        pool = SamplePool(POOL_SIZE, STATE_SIZE, EMOJI)
    else:
        pool = VideoPool(FRAME_STRIDE, SKIP_RANGE, POOL_SIZE, STATE_SIZE, VIDEO)
    return pool

def main(args):
    parser = argparse.ArgumentParser('Train Neural CA')
    parser.add_argument(
        "--run_name",
        type=str,
        default=None)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural-ca")
    parser.add_argument(
        "--train_steps",
        type=int,
        default=2000)
    parser.add_argument(
        "--pool_type",
        type=str,
        default="EMOJI")
    args = parser.parse_args(args)

    wandb.init(
        project=args.wandb_project,
        name=args.run_name
    )

    model = build_model()
    optimizer = build_optimizer()
    pool = build_pool(pool_type=args.pool_type)
    os.makedirs(os.path.join("logging", wandb.run.name), exist_ok=True)
    train(model, optimizer, args.train_steps, pool)

if __name__ == '__main__':
    main(sys.argv[1:])
