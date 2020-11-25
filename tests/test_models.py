import pytest
import numpy as np

from context import neural_ca
from neural_ca.models.automata import AutomataModel

@pytest.fixture
def example_automata_model():
    return AutomataModel(
        state_size=16,
        drop_prob=0.5
    )

@pytest.mark.models
class TestAutomataModel:
    @pytest.mark.parametrize("state_size", [4, 16, 64, 256])
    def test_build_filters_shape(self, state_size):
        model = AutomataModel(state_size=state_size)
        filters = model.build_filters().numpy()
        assert filters.shape == (3, 3, state_size, 3)

    @pytest.mark.parametrize("s", [4, 16, 64, 256])
    def test_build_stochastic_mask_shape(self, example_automata_model, s):
        cell_state = np.ones([1, s, s, example_automata_model.state_size])
        mask = example_automata_model.build_stochastic_mask(cell_state).numpy()
        assert mask.shape == (1, s, s, 1)
    
    @pytest.mark.parametrize("s", [4, 16, 64, 256])
    def test_build_live_mask_shape(self, example_automata_model, s):
        cell_state = np.ones([1, s, s, example_automata_model.state_size])
        mask = example_automata_model.build_live_mask(cell_state).numpy()
        assert mask.shape == (1, s, s, 1)

    @pytest.mark.parametrize("x, y, alive", [(0, 0, 4), (6, 6, 9), (15, 8, 6)])
    def test_build_live_mask_masking(self, example_automata_model, x, y, alive):
        cell_state = np.zeros([1, 16, 16, example_automata_model.state_size])
        cell_state[0, x, y, 3] = 1
        mask = example_automata_model.build_live_mask(cell_state).numpy()
        assert np.sum(mask[:, max(0, x - 1):x + 2, max(0, y - 1):y + 2]) == alive
        assert np.sum(mask) == alive

    def test_call_runs(self, example_automata_model):
        cell_state = np.zeros([1, 64, 64, example_automata_model.state_size])
        cell_state = example_automata_model(cell_state)
        assert True
