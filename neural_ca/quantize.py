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

NUM_EXAMPLES = 10

def save_video(model, pool):
    video = util.video.make_video(model, pool, VIDEO_STEPS)
    clip = mpy.ImageSequenceClip(video, fps=16)
    filename = os.path.join("logging", "quantize", "test.mp4")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    clip.write_videofile(filename, logger=None)


def gen_examples(model, pool):
    cell = pool.build_seeds()
    for _ in range(NUM_EXAMPLES):
        yield [cell]
        cell = model(cell)


def main(args):
    parser = argparse.ArgumentParser('Quantize Neural CA')
    parser.add_argument(
        "--load",
        type=str,
        default=None
    )
    args = parser.parse_args(args)
    assert args.load

    model = tf.keras.models.load_model(args.load)
    pool = SamplePool(POOL_SIZE, STATE_SIZE, EMOJI)
    save_video(model, pool)

    def representative_data_gen():
        yield from gen_examples(model, pool)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)


if __name__ == '__main__':
    main(sys.argv[1:])
