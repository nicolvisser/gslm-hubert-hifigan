# from hubert.hubert import HubertDiscrete
# from kmeans.kmeans import KMeansInference
# from duration.duration_predictor import DurationPredictor
# from acoustic.acoustic import AcousticModel
# from hifigan.hifigan.generator import HifiganGenerator

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from dedupe import dedupe as dedupe_fn
from dpdp import dpdp


class GSLM(nn.Module):
    def __init__(self, n_units, dp_lambda=0):
        super(GSLM, self).__init__()

        self.n_units = n_units
        self.dp_lambda = dp_lambda

        self.hubert = torch.hub.load(
            "bshall/hubert:main",
            "hubert_discrete",
            trust_repo=True,
        )

        self.kmeans = torch.hub.load(
            "nicolvisser/hubert-kmeans:main",
            "kmeans",
            features="hubert-bshall",
            dataset="librispeech",
            n_units=n_units,
            trust_repo=True,
        )

        self.duration_predictor = torch.hub.load(
            "nicolvisser/duration-predictor:main",
            "duration_predictor",
            features="hubert-bshall",
            dataset="ljspeech",
            n_units=n_units,
            dp_lmbda=dp_lambda,
            trust_repo=True,
        )

        self.acoustic = torch.hub.load(
            "nicolvisser/acoustic-model:main",
            "acoustic",
            features="hubert-bshall",
            dataset="ljspeech",
            n_units=n_units,
            dp_lmbda=dp_lambda,
            trust_repo=True,
        )

        self.hifigan = torch.hub.load(
            "bshall/hifigan:main",
            "hifigan",
        )

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, sr: int, dedupe: bool = True) -> torch.Tensor:
        assert wav.dim() == 2, "wav must be 2D"
        assert sr == 16000
        # trucate wavefore to nearest 320 frames shorter than original
        wav = wav[:, : wav.shape[1] - (wav.shape[1] % 320)]
        # pad the waveform such that labels are exactly aligned with waveform
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2), mode="reflect")
        # batch
        wav = wav.unsqueeze(0)
        # extract features
        features, _ = self.hubert.encode(wav, layer=7)
        # unbatch
        features = features.squeeze(0)

        # discretize
        if self.dp_lambda == 0:  # kmeans
            units = self.kmeans.predict(features)
        else:  # dpdp
            units = dpdp(
                features, codebook=self.kmeans.cluster_centers, lmbda=self.dp_lambda
            )

        # optionally dedupe
        if dedupe:
            units = dedupe_fn(units)

        return units

    @torch.inference_mode()
    def decode(
        self, units: torch.Tensor, deduped: bool = True
    ) -> Tuple[torch.Tensor, int]:
        if deduped:
            units = self.duration_predictor.redupe(units)

        # batch
        units = units.unsqueeze(0)
        # generate mels
        mels = self.acoustic.generate(units)
        # transpose last two dimensions
        mels = mels.permute(0, 2, 1)
        # generate audio
        audio = self.hifigan(mels)
        # unbatch
        audio = audio.squeeze(0)

        return audio, 16000

    @torch.inference_mode()
    def dedupe(self, units: torch.Tensor) -> torch.Tensor:
        return dedupe_fn(units)

    @torch.inference_mode()
    def redupe(self, units: torch.Tensor) -> torch.Tensor:
        return self.duration_predictor.redupe(units)
