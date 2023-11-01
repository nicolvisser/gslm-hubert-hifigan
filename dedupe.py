import torch


@torch.jit.script
def dedupe(x: torch.Tensor):
    diffs = torch.cat(
        (x[:-1] != x[1:], torch.tensor([True], dtype=torch.bool, device=x.device))
    )
    return x[diffs]
