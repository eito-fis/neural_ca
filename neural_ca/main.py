import numpy as np
import tensorflow as tf

import util
from models.automata import AutomataModel

### CONSTANTS ###
HEIGHT = 64
WIDTH = 64
BATCH_SIZE = 4
STATE_SIZE = 16
DROP_PROB = 0.5
TRAIN_STEPS = 100
GEN_RANGE = (64, 96)
EMOJI = "🦎"
LR = 2e-3

def build_model():
    model = AutomataModel(STATE_SIZE, drop_prob=DROP_PROB)
    return model

def build_optimizer():
    lr_scheduler = tf.optimizers.schedules.PiecewiseConstantDecay(
        [2000], [LR, LR * 0.1]
    )
    optimizer = tf.keras.optmizers.Adam(lr_sched)
    return optimizer

def calc_loss(cells, image):
    pixel_delta = util.to_rgba(cells) - image
    loss = tf.reduce_mean(tf.squared(pixel_delta))
    return loss

def train(model, optimizer, train_steps, image):
    for i in tf.range(TRAIN_STEPS):
        cells = util.make_seeds(image.shape, BATCH_SIZE, STATE_SIZE)
        gen_steps = tf.random.uniform([], GEN_RANGE[0], GEN_RANGE[1], tf.int32)
        with tf.GradientTape() as tape:
            for j in tf.range(steps):
                cells = model(cells)
            loss = calc_loss(cells, image)
        grads = tape.gradient(loss, model.trainable_weights)
        grads = [g/(tf.norm(g) + 1e-8) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

def main():
    model = build_model()
    optimizer = build_optimizer()
    image = util.load_emoji(EMOJI)
    train(model, optimizer, image)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Train PPO')

    # parser.add_argument(
    #     "--actor_weights",
    #     type=str,
    #     default=None)
    # parser.add_argument(
    #     "--env",
    #     type=str,
    #     default="BipedalWalker-v3")
    # parser.add_argument(
    #     '--gpu',
    #     default=False,
    #     action='store_true')
    # args = parser.parse_args()

    #logging.getLogger().setLevel(logging.INFO)
    # main(args)
    main()

