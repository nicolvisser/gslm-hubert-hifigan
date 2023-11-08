from pathlib import Path

import click
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from gslm import GSLM


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("-n", "--n_units", type=click.INT, default=500)
@click.option("-l", "--dp_lambda", type=click.FLOAT, default=0.0)
@click.option("-e", "--extension", default=".txt")
def resynthesize(input, output, n_units, dp_lambda, extension):
    """Decodes a directory of text files into a audio files"""

    input = Path(input)
    output = Path(output)

    click.echo(f"Decoding text files from {input.absolute()}")
    click.echo(f"Saving encoded audio files to {output.absolute()}")
    click.echo(f"Using {n_units} units and {dp_lambda} lambda")
    click.echo("Loading model...")

    model = GSLM(n_units=n_units, dp_lambda=dp_lambda).cuda().eval()

    for txt_path in tqdm(sorted(list(input.rglob(f"*{extension}")))):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                unicode_text = f.read()

            unit_values = [ord(char) - 0x4E00 for char in unicode_text]

            unit_array = np.array(unit_values, dtype=np.int64)

            units = torch.from_numpy(unit_array).cuda()

            wav, sr = model.decode(units, deduped=True)

            out_path = output / txt_path.relative_to(input).with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(out_path, wav.cpu(), sr)

        except Exception as e:
            click.echo(f"Failed to encode {txt_path}")
            (output / "errors.txt").write_text(f"{txt_path}\n{e}\n\n")


if __name__ == "__main__":
    resynthesize()
