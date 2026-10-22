import numpy as np
import pytest
import torch

from gully.student.heads import HEADS
from gully.student.train import train_student


@pytest.mark.parametrize("head", list(HEADS))
def test_heads_forward_and_mask(head):
    x = torch.randn(4, 8, 65, 384)
    v = torch.ones(4, 8, dtype=torch.bool)
    v[:, 7] = False
    m = HEADS[head]().eval()
    assert m(x, v).shape == (4, 2)


def test_train_learns_separable_toy():
    r = np.random.default_rng(0)
    y = r.integers(0, 2, 200)
    X = r.normal(size=(200, 8, 65, 16)).astype(np.float16)
    X[:, :, 0, 0] += (2 * y - 1)[:, None] * 3  # signal in the CLS token
    V = np.ones((200, 8), dtype=bool)
    _, hist, _, ep = train_student(
        X[:150], V[:150], y[:150].astype(float), X[150:], V[150:], y[150:], head="mlp",
        epochs=8, bs=32, device="cpu", head_kwargs={"dim": 16, "hidden": 32})
    assert hist[ep]["acc"] > 0.85
