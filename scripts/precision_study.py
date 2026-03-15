"""Precision study: does computing the correlation / model in fp16 or bf16 move MACE, and is it
worth it for throughput. fp8 is assessed honestly rather than forced.

Writes docs/precision.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from cuda_motion_flow.learned.correlation import local_correlation_reference
from cuda_motion_flow.learned.data import SyntheticPairGenerator, load_image_pool
from cuda_motion_flow.learned.losses import mace
from cuda_motion_flow.learned.model import IHN

OUT = Path("docs")
WEIGHTS = Path("weights/ihn.pt")


def cuda_time_ms(fn, reps: int = 200, warmup: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def _time_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    return cuda_time_ms(lambda: local_correlation_reference(a, b, 4))


def correlation_throughput() -> dict[str, float]:
    """Reference correlation timed in fp32 vs fp16 vs bf16 at the IHN's 16x16 shape."""
    fa = torch.randn(32, 96, 16, 16, device="cuda")
    fb = torch.randn(32, 96, 16, 16, device="cuda")
    out = {}
    for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        out[f"corr_ref_{name}_ms"] = round(_time_corr(fa.to(dt), fb.to(dt)), 4)
    ref = local_correlation_reference(fa, fb, 4)
    half = local_correlation_reference(fa.half(), fb.half(), 4).float()
    out["corr_fp16_rel_err"] = round(float((half - ref).abs().max() / ref.abs().max()), 5)
    return out


@torch.no_grad()
def mace_by_precision(model: IHN, val) -> dict[str, float]:
    pa, pb, delta = val
    res = {}
    model_out = model(pa, pb)
    res["mace_fp32"] = round(float(mace(model_out[-1], delta)), 4)
    for name, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        with torch.autocast(device_type="cuda", dtype=dt):
            out = model(pa, pb)
        res[f"mace_{name}"] = round(float(mace(out[-1].float(), delta)), 4)
        res[f"mace_{name}_delta"] = round(res[f"mace_{name}"] - res["mace_fp32"], 4)
    return res


def fp8_assessment() -> dict[str, object]:
    has_fp8 = hasattr(torch, "float8_e4m3fn")
    has_scaled_mm = hasattr(torch, "_scaled_mm")
    return {
        "fp8_dtype_available": has_fp8,
        "scaled_mm_available": has_scaled_mm,
        "note": (
            "torch exposes fp8 and _scaled_mm, but the cost volume at 16x16 is a tiny per-pair "
            "dot product, not a large GEMM, so an fp8 tensor-core path has nothing to amortise. "
            "the kernel is latency bound (see optimization_log), so fp8 would be marginal here "
            "and was not pursued."
        ),
    }


def main() -> None:
    torch.manual_seed(0)
    model = IHN(size=128, iters=6).cuda().eval()
    if WEIGHTS.exists():
        model.load_state_dict(torch.load(str(WEIGHTS), map_location="cuda"))

    pool = load_image_pool("data/coco", size=(240, 320), limit=200)
    gen = SyntheticPairGenerator(pool, patch=128, rho=32, seed=7)
    val = gen.sample(64)

    result = {
        "throughput": correlation_throughput(),
        "accuracy": mace_by_precision(model, val),
        "fp8": fp8_assessment(),
        "weights_loaded": WEIGHTS.exists(),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "precision.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
