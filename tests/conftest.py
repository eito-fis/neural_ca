import pytest

from neural_ca.pools import SamplePool

@pytest.fixture
def example_emoji():
    return "🦎"

@pytest.fixture
def make_pool(example_emoji):
    def _func(pool_size, state_size):
        return SamplePool(
            pool_size=pool_size,
            state_size=state_size,
            data=example_emoji,
        )
    return _func
