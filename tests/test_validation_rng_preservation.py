import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))

from diffusion_checkpoint import preserve_rng_state


def _draw():
    values = [random.random(), float(np.random.rand()), float(torch.rand(()))]
    if torch.cuda.is_available():
        values.append(float(torch.rand((), device="cuda")))
    return values


def test_validation_preserves_python_numpy_torch_and_cuda_rng():
    random.seed(71); np.random.seed(71); torch.manual_seed(71)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(71)
    _draw()
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    with preserve_rng_state(999):
        _draw(); _draw()
    actual = _draw()
    random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
    if states[3] is not None: torch.cuda.set_rng_state_all(states[3])
    expected = _draw()
    assert actual == expected


def test_validation_seed_is_reproducible_without_leaking():
    with preserve_rng_state(13): first = _draw()
    with preserve_rng_state(13): second = _draw()
    assert first == second
