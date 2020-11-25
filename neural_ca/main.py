import numpy as np
import tensorflow as tf

from models.automata import AutomataModel

### CONSTANTS ###
HEIGHT = 64
WIDTH = 64
STATE_SIZE = 16
DROP_PROB = 0.5
TRAIN_STEPS = 100
GEN_STEPS = 100

def build_cell():
    """ Build initial cell state """
    cell_state = np.zeros([HEIGHT, WIDTH, STATE_SIZE])
    cell_state[HEIGHT // 2, WIDTH // 2, 3:] += 1
    cell_state = tf.convert_to_tensor(cell_state)
    return cell_state

def build_model():
    """ Build automatat model """
    model = AutomataModel(HEIGHT, WIDTH, STATE_SIZE, drop_prob=DROP_PROB)
    return model

def train(model, train_steps):
    for i in train_steps:
        cell = build_cell()
        for g in GEN_STEPS:
            cell = model(cell)
        update_model(model, cell_state)

def update_model(model):
    pass

def main():
    model = build_model()
    train(model, TRAIN_STEPS)

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

