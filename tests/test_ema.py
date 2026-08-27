"""EMA warm-up.

With gradient accumulation a short run performs very few EMA updates. At a fixed
decay of 0.999, 350 updates leave 0.999**350 = 0.70 of the shadow sitting on the
random initialisation -- and inference prefers the shadow.
"""

import torch
from torch import nn

from ct_restore.training import EMA


def _model(value: float) -> nn.Module:
    model = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def _weight(module: nn.Module) -> float:
    return float(next(module.parameters()).mean())


def test_warmup_tracks_the_model_early() -> None:
    ema = EMA(_model(0.0), decay=0.999)
    target = _model(1.0)
    for _ in range(20):
        ema.update(target)
    # A fixed 0.999 decay would leave the shadow at ~0.02 after 20 updates.
    assert _weight(ema.shadow) > 0.5


def test_without_warmup_a_short_run_stays_near_initialisation() -> None:
    ema = EMA(_model(0.0), decay=0.999, warmup=False)
    target = _model(1.0)
    for _ in range(20):
        ema.update(target)
    assert _weight(ema.shadow) < 0.05


def test_effective_decay_rises_toward_the_configured_value() -> None:
    ema = EMA(_model(0.0), decay=0.999)
    assert ema.effective_decay() < 0.2
    ema.updates = 100
    early = ema.effective_decay()
    ema.updates = 100_000
    assert early < ema.effective_decay() == 0.999


def test_update_counter_advances() -> None:
    ema = EMA(_model(0.0), decay=0.9)
    for _ in range(5):
        ema.update(_model(1.0))
    assert ema.updates == 5


def test_shadow_converges_to_the_model_given_enough_updates() -> None:
    ema = EMA(_model(0.0), decay=0.9)
    target = _model(1.0)
    for _ in range(500):
        ema.update(target)
    assert abs(_weight(ema.shadow) - 1.0) < 1e-3
