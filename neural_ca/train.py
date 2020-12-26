import wandb
import numpy as np
import tensorflow as tf

from neural_ca import util
from neural_ca.models.automata import AutomataModel

### CONSTANTS ###
HEIGHT = 64
WIDTH = 64
BATCH_SIZE = 4
STATE_SIZE = 16
DROP_PROB = 0.5
VAL_STEPS = 250
VIDEO_STEPS = 256
GEN_RANGE = (64, 96)
EMOJI = "🦎"
LR = 2e-3

def make_video(model, image):
    def process_cell(cell):
        # Remove batch dimension
        np_cell = np.asarray(cell)[0, :, :, :]
        # From channel last to channel first for wandb
        swap_cell = np.moveaxis(np_cell, 0, 2)
        return swap_cell

    cell = util.make_seeds(image.shape, 1, STATE_SIZE)
    video = [process_cell(cell)]
    for i in range(VIDEO_STEPS):
        cell = model(cell)
        video.append(process_cell(cell))

def log(i, loss, grads, model, image):
    log_data = {
        "step": i,
        "loss": loss,
        "grads": wandb.Histogram(grads)
    }
    if i % VAL_STEPS == 0:
        video = make_video(model, image)
        log_data["video"] = wandb.Video(video)
    wandb.log(log_data)

def calc_loss(cells, image):
    pixel_delta = util.to_rgba(cells) - image
    loss = tf.reduce_mean(tf.square(pixel_delta))
    return loss

def train(model, optimizer, train_steps, image):
    for i in tf.range(train_steps):
        cells = util.make_seeds(image.shape, BATCH_SIZE, STATE_SIZE)
        gen_steps = tf.random.uniform([], GEN_RANGE[0], GEN_RANGE[1], tf.int32)
        with tf.GradientTape() as tape:
            for j in tf.range(gen_steps):
                cells = model(cells)
            loss = calc_loss(cells, image)
        grads = tape.gradient(loss, model.trainable_weights)
        grads = [g/(tf.norm(g) + 1e-8) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        log(i, loss, grads, model, image)

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
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name
    )

    model = build_model()
    optimizer = build_optimizer()
    image = util.load_emoji(EMOJI)
    train(model, optimizer, args.train_steps, image)

if __name__ == '__main__':
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
        default=8000)

    main(args)

