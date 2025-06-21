import pytest

import test_functions
from template_nn.optimisers.hho import HHO


@pytest.fixture
def setup_hho():
    func_details = test_functions.getFunctionDetails(0)

    config = {
        "objective_function": getattr(test_functions, func_details[0]),
        "lower_bound": func_details[1],
        "upper_bound": func_details[2],
        "dimension": func_details[3],
        "search_agents_num": 100,
        "max_iterations": 100,
    }

    hho = HHO(config)
    return hho


def test_hho_initialisation(setup_hho):
    hho = setup_hho

    assert hho.objective_function == getattr(test_functions, "F1")
    assert hho.lower_bound == -100
    assert hho.upper_bound == 100
    assert hho.dimension == 30
    assert hho.search_agents_num == 100
    assert hho.max_iterations == 100

    print(hho)
