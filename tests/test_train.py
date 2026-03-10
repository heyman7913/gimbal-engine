import math

import pytest

pytestmark = pytest.mark.cuda


def test_supervised_training_step_runs():
    import torch

    from cuda_motion_flow.learned.data import SyntheticPairGenerator
    from cuda_motion_flow.learned.model import IHN
    from cuda_motion_flow.learned.train import PhaseAConfig, train_supervised

    pool = (torch.rand(32, 120, 160) * 255).to(torch.uint8)
    gen = SyntheticPairGenerator(pool, patch=64, rho=12, margin=8)
    model = IHN(size=64, iters=3).cuda()
    val = gen.sample(8)
    cfg = PhaseAConfig(
        steps=30, batch=8, eval_every=15, patience=100, time_budget_s=120, log=lambda _s: None
    )
    history = train_supervised(model, gen, val, cfg)
    assert len(history.val_mace) >= 1
    assert math.isfinite(history.best_mace)


def test_unsupervised_training_step_runs():
    import torch

    from cuda_motion_flow.learned.model import IHN
    from cuda_motion_flow.learned.train import PhaseBConfig, train_unsupervised

    model = IHN(size=64, iters=3).cuda()
    pairs = torch.rand(16, 2, 64, 64)
    cfg = PhaseBConfig(steps=5, batch=4, log=lambda _s: None)
    losses = train_unsupervised(model, pairs, cfg)
    assert len(losses) == 5
    assert all(math.isfinite(x) for x in losses)
