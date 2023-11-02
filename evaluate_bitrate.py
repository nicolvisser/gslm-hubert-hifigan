from pathlib import Path
import numpy as np

import torchaudio
from config import ExperimentConfig

N_UNITS = 50
DP_LAMBDA = 0

units_deduped_dir = (
    Path(ExperimentConfig.output_dir)
    / f"n_units-{N_UNITS}-dp_lambda-{DP_LAMBDA}"
    / "units_deduped"
)

librispeech_dir = Path(ExperimentConfig.librispeech_dir)

total_num_units = 0
total_wav_duration = 0

for units_deduped_path in units_deduped_dir.rglob("*.npy"):
    relative_path = units_deduped_path.relative_to(units_deduped_dir)

    wav_original_path = librispeech_dir / relative_path.with_suffix(".flac")

    torchinfo = torchaudio.info(wav_original_path)

    wav_duration = torchinfo.num_frames / torchinfo.sample_rate

    num_units = len(np.load(units_deduped_path))

    total_num_units += num_units
    total_wav_duration += wav_duration


average_units_duration = total_wav_duration / total_num_units
bits_per_unit = np.log2(N_UNITS)  # deliberately not using np.ceil here
bitrate = bits_per_unit / average_units_duration


print(f"Total number of units: {total_num_units} units")
print(f"Total duration of audio: {total_wav_duration} seconds")
print(f"Average duration of units: {average_units_duration} seconds")
print(f"Bit Rate: {bitrate} bits/sec")
