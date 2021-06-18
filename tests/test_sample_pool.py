import pytest
import tensorflow as tf

from neural_ca.pools import SamplePool

@pytest.fixture
def make_pool(example_emoji):
    def _func(pool_size, state_size):
        return SamplePool(
            pool_size=pool_size,
            state_size=state_size,
            emoji=example_emoji,
        )
    return _func

@pytest.mark.pool
class TestSamplePool:
    # TODO: Add shape back in once config is created
    @pytest.mark.parametrize("pool_size", [16, 32])
    @pytest.mark.parametrize("state_size", [16, 32])
    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_sample_shape(self, make_pool, pool_size, state_size, batch_size):
        pool = make_pool(pool_size, state_size)
        sample, _, _ = pool.sample(batch_size)
        sample = sample.numpy()
        x, y, *_ = pool.shape
        assert sample.shape == (batch_size, x, y, state_size)

    def test_sample_batch_size_error(self, make_pool):
        pool = make_pool(16, 16)
        with pytest.raises(AssertionError):
            pool.sample(100)

    @pytest.mark.parametrize("state_size", [16, 32])
    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_build_seeds(self, make_pool, batch_size, state_size):
        pool = make_pool(32, state_size)
        seeds = pool.build_seeds(batch_size)
        assert tf.reduce_sum(seeds).numpy() == batch_size * (state_size - 3)

    def test_sample_fresh_seed(self, make_pool):
        state_size = 16
        pool = make_pool(16, state_size)
        pool.pool = tf.ones(pool.pool.shape)

        sample, _, _ = pool.sample(1)
        fresh_seed = pool.build_seeds(1)

        assert sample.shape == fresh_seed.shape
        assert tf.math.reduce_all(tf.math.equal(sample, fresh_seed))

    @pytest.mark.parametrize("num_idx", [1, 4, 8])
    def test_update(self, num_idx, make_pool):
        state_size = 16
        pool = make_pool(16, state_size)

        sample = tf.ones(pool.pool.shape)[:num_idx]
        idxs = tf.range(0, num_idx)
        pool.update(sample, idxs)
        sample_in_pool = tf.gather(pool.pool, idxs, axis=0)

        assert sample.shape == sample_in_pool.shape
        assert tf.math.reduce_all(tf.math.equal(sample, sample_in_pool))

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_sample_update(self, batch_size, make_pool):
        state_size = 16
        pool = make_pool(16, state_size)
        sample, _, idxs = pool.sample(batch_size)

        sample = tf.ones(sample.shape)
        pool.update(sample, idxs)
        sample_in_pool = tf.gather(pool.pool, idxs, axis=0)

        assert sample.shape == sample_in_pool.shape
        assert tf.math.reduce_all(tf.math.equal(sample, sample_in_pool))
