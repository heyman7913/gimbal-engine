import math

import pytest

pytestmark = pytest.mark.cuda


def test_supervised_training_step_runs():
    import torch
    from gimbal.learned.data import SyntheticPairGenerator
    from gimbal.learned.model import IHN
    from gimbal.learned.train import PhaseAConfig, train_supervised

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
    from gimbal.learned.model import IHN
    from gimbal.learned.train import PhaseBConfig, train_unsupervised

    model = IHN(size=64, iters=3).cuda()
    pairs = torch.rand(16, 2, 64, 64)
    cfg = PhaseBConfig(steps=5, batch=4, log=lambda _s: None)
    losses = train_unsupervised(model, pairs, cfg)
    assert len(losses) == 5
    assert all(math.isfinite(x) for x in losses)


def test_phase_b_guard_reverts_a_degrading_run():
    import copy

    import torch
    from gimbal.learned.model import IHN
    from gimbal.learned.train import PhaseBConfig, phase_b_with_guard

    torch.manual_seed(0)
    model = IHN(size=64, iters=2).cuda()
    pairs = torch.rand(8, 2, 64, 64)
    before = copy.deepcopy(model.state_dict())

    # held-out metric (higher is better) drops after fine-tuning, so the run must be rejected
    calls = {"n": 0}

    def degrading(_m: IHN) -> float:
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else 0.0

    cfg = PhaseBConfig(steps=3, batch=4, log=lambda _s: None)
    result = phase_b_with_guard(model, pairs, degrading, cfg)

    assert not result["adopted"]
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
