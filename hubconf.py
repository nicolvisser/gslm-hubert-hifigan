dependencies = ["torch", "torchaudio", "sklearn", "matplotlib"]
from gslm import GSLM  # noqa: E402


def gslm(n_units: int = 500, dp_lambda: int = 0) -> GSLM:
    return GSLM(n_units=n_units, dp_lambda=dp_lambda)
