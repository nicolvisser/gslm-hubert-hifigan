from pathlib import Path

import click
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from gslm import GSLM

dp_lmbdas = {
    50: [0],
    100: [0, 4, 8, 12, 16, 20],
    200: [0, 4, 8, 12, 16, 20],
    500: [0, 4, 8, 12, 16, 20, 24, 28],
    1000: [0, 4, 8, 12, 16, 20],
    2000: [0, 4, 8, 12, 16, 20],
}


@click.command()
@click.argument("librispeech_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("-s", "--subset", default="dev-clean")
def run_experiment(librispeech_dir, output_dir, subset):
    """Resynthesize a directory of audio files."""

    librispeech_dir = Path(librispeech_dir)
    output_dir = Path(output_dir)

    dataset = torchaudio.datasets.LIBRISPEECH(
        librispeech_dir.parent,
        url=subset,
        download=False,
        folder_in_archive=librispeech_dir.name,
    )

    for n_units in dp_lmbdas.keys():
        for dp_lambda in dp_lmbdas[n_units]:
            click.echo(f"Running for {n_units} units and lambda={dp_lambda}")
            click.echo("Loading model...")

            model = GSLM(n_units=50, dp_lambda=0).cuda().eval()

            output_subdir = output_dir / f"n_units-{n_units}-dp_lambda-{dp_lambda}"

            if output_subdir.exists():
                continue

            for i, (wav, sr, *_) in enumerate(tqdm(dataset)):
                try:
                    relative_path, *_ = dataset.get_metadata(i)

                    wav = wav.cuda()
                    units = model.encode(wav, sr, dedupe=False)
                    units_deduped = model.dedupe(units)
                    units_ = model.redupe(units_deduped)
                    wav_, sr = model.decode(units_, deduped=False)

                    units_out_path = (
                        output_subdir / "units" / relative_path
                    ).with_suffix(".npy")
                    units_deduped_out_path = (
                        output_subdir / "units_deduped" / relative_path
                    ).with_suffix(".npy")
                    units_reduped_out_path = (
                        output_subdir / "units_reduped" / relative_path
                    ).with_suffix(".npy")
                    wav_resynthesized_out_path = (
                        output_subdir / "wav_resynthesized" / relative_path
                    ).with_suffix(".wav")

                    units_out_path.parent.mkdir(parents=True, exist_ok=True)
                    units_deduped_out_path.parent.mkdir(parents=True, exist_ok=True)
                    units_reduped_out_path.parent.mkdir(parents=True, exist_ok=True)
                    wav_resynthesized_out_path.parent.mkdir(parents=True, exist_ok=True)

                    units = units.to(dtype=torch.int16, device="cpu")
                    units_deduped = units_deduped.to(dtype=torch.int16, device="cpu")
                    units_ = units_.to(dtype=torch.int16, device="cpu")
                    wav_ = wav_.to(device="cpu")

                    np.save(units_out_path, units.cpu().numpy())
                    np.save(units_deduped_out_path, units_deduped.cpu().numpy())
                    np.save(units_reduped_out_path, units_.cpu().numpy())
                    torchaudio.save(wav_resynthesized_out_path, wav_, sr)

                except Exception as e:
                    click.echo(f"Failed to resynthesize {relative_path}")
                    (output_subdir / "errors.txt").write_text(
                        f"{relative_path}\n{e}\n\n"
                    )


if __name__ == "__main__":
    run_experiment()