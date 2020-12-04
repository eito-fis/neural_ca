import pytest
import numpy as np

from context import neural_ca
from neural_ca.util import load_image_from_url, process_image

@pytest.fixture
def example_url():
    return "https://raw.githubusercontent.com/googlefonts/noto-emoji/master/png/128/emoji_u1f98e.png"

@pytest.fixture
def example_image(example_url):
    return load_image_from_url(example_url)

@pytest.fixture
def example_processed_image(example_image, s):
    return process_image(example_image, s)

@pytest.mark.utils
class TestImages:
    def test_load_image_from_url(self, example_url):
        load_image_from_url(example_url)
        assert True

    @pytest.mark.parametrize("example_processed_image, s",
                             [(4, 4), (16, 16), (64, 64)],
                             indirect=["example_processed_image"])
    def test_process_image_size(self, example_processed_image, s):
        assert example_processed_image.shape == (s, s, 4)
       
    # This is dumb but nothing else works
    @pytest.mark.parametrize("example_processed_image, s",
                             [(4, 4), (16, 16), (64, 64)],
                             indirect=["example_processed_image"])
    def test_process_image_normalize(self, example_processed_image, s):
        for r in example_processed_image:
            for c in r:
                for cell in c:
                    assert cell >= 0 and cell <= 1
