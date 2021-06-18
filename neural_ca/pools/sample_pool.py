import numpy as np
import tensorflow as tf

from neural_ca import util

class SamplePool:
    def __init__(self, pool_size, state_size, data):
        assert pool_size > 0 and state_size > 0
        self.pool_size = pool_size
        self.state_size = state_size
        self.data = self.build_data(data)
        self.shape = self.build_shape()
        self.pool = self.build_pool()

    def sample(self, batch_size):
        sample, sample_idxs = self.build_sample(batch_size)

        # Replace one sampled seed with a fresh seed
        fresh_seed = self.build_seeds(1)
        sample = tf.tensor_scatter_nd_update(
            sample,
            tf.constant([[0]]),
            fresh_seed,
        )

        target = self.build_target()

        return sample, target, sample_idxs

    def update(self, new_seeds, idxs):
        self.pool = tf.tensor_scatter_nd_update(
            self.pool,
            tf.expand_dims(idxs, axis=-1),
            new_seeds,
        )

    def build_data(self, data):
        return util.image.load_emoji(data)

    def build_shape(self):
        return self.data.shape

    def build_pool(self):
        return self.build_seeds(self.pool_size)

    def build_sample(self, batch_size):
        assert batch_size <= self.pool_size, "Batch size larger than pool size."
        idxs = tf.range(0, self.pool_size)
        sample_idxs = tf.random.shuffle(idxs)[:batch_size]
        sample = tf.gather(self.pool, sample_idxs, axis=0)
        return sample, sample_idxs

    def build_target(self):
        return self.data

    def build_seeds(self, batch_size):
        """ Makes batch of seeds """
        height, width = self.shape[:2]
        seeds = np.zeros([batch_size, height, width, self.state_size])
        seeds[:, height // 2, width // 2, 3:] += 1
        seeds = tf.convert_to_tensor(seeds, dtype=tf.float32)
        return seeds
