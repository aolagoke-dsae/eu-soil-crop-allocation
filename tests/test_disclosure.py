"""Disclosure control: published tables carry no single point's soil value."""

import numpy as np
import pandas as pd
import pytest

from eusoil.config import load_config
from eusoil.interpret import ale_grid


def test_ale_grid_never_publishes_the_minimum_or_maximum():
    values = pd.Series(np.random.default_rng(0).lognormal(size=2000))
    grid = ale_grid(values, 20, load_config()["interpretation"]["ale_quantile_range"])
    assert values.min() < grid.min() and grid.max() < values.max()
    assert grid[0] == pytest.approx(values.quantile(0.01))
    assert grid[-1] == pytest.approx(values.quantile(0.99))


@pytest.mark.parametrize("bad", [[0.0, 0.99], [0.01, 1.0], [0.5, 0.4]])
def test_ale_grid_refuses_a_range_that_reaches_the_extremes(bad):
    with pytest.raises(ValueError):
        ale_grid(pd.Series(np.arange(100.0)), 10, bad)


def test_distribution_figures_drop_each_extreme():
    from eusoil.figures import inner_values

    values = np.arange(1000.0)
    inner = inner_values(values)
    assert values.min() not in inner and values.max() not in inner
    assert inner.min() >= np.quantile(values, 0.01) and inner.max() <= np.quantile(values, 0.99)
