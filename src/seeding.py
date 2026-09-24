"""Seeding.

Seeds everywhere, and log every seed.

There are four independent sources of randomness in this project and they must
all be nailed down, because a probe AUROC that moves by 0.02 between runs is
indistinguishable from a real quantization effect of the same size:

1. Python's `random`      - dataset shuffling, topic split assignment.
2. NumPy                  - array shuffles, sklearn's internal RNG draws.
3. PyTorch                - dropout (off at eval, but also init of any
                            untrained head) and, more importantly, the order
                            of non-deterministic reductions on GPU.
4. sklearn's LogisticRegression - takes its own `random_state`; it does NOT
                            read the global NumPy seed for the solvers we use.
                            This is passed explicitly in probes/, not here.

`PYTHONHASHSEED` is set too, but note it only takes effect for a *fresh*
interpreter - setting it inside a running process does not retroactively
change str hashing. It is set here so that subprocesses inherit it, and the
value is logged so a reader knows what to export for an exact replay.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed every RNG we touch. Returns a record of what was set, for logging.

    Args:
        seed: the master seed.
        deterministic_torch: if True, ask cuDNN/cuBLAS for deterministic
            kernels. This costs speed. We keep it on because extraction is
            run once and cached - a 20% slowdown on a 30-minute job is a fair
            price for bit-identical reruns, and reproducibility is the whole
            point of a measurement paper.

    Returns:
        dict describing the seeding, suitable for dumping into a JSON sidecar.
    """
    record: dict[str, Any] = {"master_seed": seed}

    os.environ["PYTHONHASHSEED"] = str(seed)
    record["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    record["python_random"] = seed

    np.random.seed(seed)
    record["numpy"] = seed

    try:
        import torch
    except ImportError:
        record["torch"] = "not installed"
        return record

    torch.manual_seed(seed)
    record["torch_cpu"] = seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        record["torch_cuda"] = seed
        record["cuda_device_count"] = torch.cuda.device_count()
    else:
        record["torch_cuda"] = "no cuda"

    if deterministic_torch:
        # cuBLAS needs this env var set before the first CUDA call to make
        # matmul reductions deterministic; setting it late is a silent no-op,
        # which is why it is here at seed time rather than at model load.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            record["deterministic_algorithms"] = "on (warn_only)"
        except Exception as exc:  # pragma: no cover - version dependent
            record["deterministic_algorithms"] = f"unavailable: {exc}"

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            record["cudnn_deterministic"] = True

    return record
