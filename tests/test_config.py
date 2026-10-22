from gully.config import load_config
from gully.seed import seed_everything


def test_data_config_has_pool_and_test():
    cfg = load_config("data")
    assert "pool" in cfg and "test" in cfg


def test_seed_everything_is_deterministic():
    import numpy as np

    seed_everything(0)
    a = np.random.rand(3)
    seed_everything(0)
    assert (a == np.random.rand(3)).all()
