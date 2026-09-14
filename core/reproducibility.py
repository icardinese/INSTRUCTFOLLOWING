"""Seeds every source of randomness this project uses. Call once at the top of any script whose
output needs to be reproducible run-to-run -- which, for a paper, is every training script.
Without this, gate/direction initialization noise alone has been observed to shift baseline_dev_mse
by +/-0.1-0.2 across "identical" reruns.
"""
import random

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
