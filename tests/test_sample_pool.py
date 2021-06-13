import pytest
import tensorflow as tf

from context import neural_ca
from neural_ca import util
from neural_ca.sample_pool import SamplePool

@pytest.fixture
def make_pool():
    def _func(pool_size, shape, state_size):
        return SamplePool(
            pool_size=pool_size,
            shape=shape,
            state_size=state_size,
        )
    return _func

@pytest.mark.pool
class TestSamplePool:
    @pytest.mark.parametrize("pool_size", [16, 32])
    @pytest.mark.parametrize("shape", [(64, 64, 3), (128, 128, 3)])
    @pytest.mark.parametrize("state_size", [16, 32])
    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_sample_shape(self, make_pool, pool_size, shape, state_size,
                          batch_size):
        pool = make_pool(pool_size, shape, state_size)
        sample, _ = pool.sample(batch_size)
        sample = sample.numpy()
        x, y, _ = shape
        assert sample.shape == (batch_size, x, y, state_size)

    def test_sample_batch_size_error(self, make_pool):
        pool = make_pool(16, (64, 64, 3), 16)
        with pytest.raises(AssertionError):
            pool.sample(100)

    def test_sample_fresh_seed(self, make_pool):
        shape, state_size = (64, 64, 3), 16
        pool = make_pool(16, shape, state_size)
        pool.pool = tf.ones(pool.pool.shape)

        sample, _ = pool.sample(1)
        fresh_seed = util.make_seeds(shape, 1, state_size)

        assert sample.shape == fresh_seed.shape
        assert tf.math.reduce_all(tf.math.equal(sample, fresh_seed))

    @pytest.mark.parametrize("num_idx", [1, 4, 8])
    def test_update(self, num_idx, make_pool):
        shape, state_size = (64, 64, 3), 16
        pool = make_pool(16, shape, state_size)

        sample = tf.ones(pool.pool.shape)[:num_idx]
        idxs = tf.range(0, num_idx)
        pool.update(sample, idxs)
        sample_in_pool = tf.gather(pool.pool, idxs, axis=0)

        assert sample.shape == sample_in_pool.shape
        assert tf.math.reduce_all(tf.math.equal(sample, sample_in_pool))

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_sample_update(self, batch_size, make_pool):
        shape, state_size = (64, 64, 3), 16
        pool = make_pool(16, shape, state_size)
        sample, idxs = pool.sample(batch_size)

        sample = tf.ones(sample.shape)
        pool.update(sample, idxs)
        sample_in_pool = tf.gather(pool.pool, idxs, axis=0)

        assert sample.shape == sample_in_pool.shape
        assert tf.math.reduce_all(tf.math.equal(sample, sample_in_pool))
