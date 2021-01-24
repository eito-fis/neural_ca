from neural_ca import util

import tensorflow as tf

class SamplePool:
    def __input__(self, pool_size, shape, state_size):
        self.shape = image.shape
        self.pool = util.make_seeds(self.shape, pool_size, state_size)
        self.state_size = state_size
        self.pool_size = pool_size

    def sample(self, batch_size):
        assert batch_size <= self.pool_size, "Batch size larger than pool size"
        l = tf.shape(self.pool)[0]
        idxs = tf.range(0, l)
        sample_idxs = tf.random.shuffle(idxs)[:batch_size]
        sample = tf.gather(self.pool, sample_idxs, axis=0)

        # Replace one sampled seed with a fresh seed
        fresh = util.make_seeds(self.shape, 1, self.state_size)
        sample = tf.tensor_scatter_nd_update(
            sample,
            tf.constant([[0]]),
            fresh_seed,
        )

        return sample, idxs

    def update(self, new_seeds, idxs):
        self.pool = tf.tensor_scatter_nd_update(
            self.pool,
            tf.expand_dims(idxs, axis=-1),
            new_seeds,
        )
