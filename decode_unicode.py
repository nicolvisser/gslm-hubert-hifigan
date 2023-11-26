from pathlib import Path

import click
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from gslm import GSLM


@click.command()
@click.option(
    "-i",
    "--input",
    help="Path to input directory containing audio.",
    type=click.Path(exists=True),
    prompt=True,
)
@click.option(
    "-o",
    "--output",
    help="path to output directory to hold text files.",
    type=click.Path(),
    prompt=True,
)
@click.option(
    "-n",
    "--n_units",
    help="The number of k-means units to use.",
    type=click.INT,
    default=500,
    prompt=True,
)
@click.option(
    "-l",
    "--dp_lambda",
    help="The lambda paramter of DPDP to use.",
    type=click.FLOAT,
    default=0.0,
    prompt=True,
)
@click.option(
    "-d",
    "--deduped",
    help="Whether the units are deduped.",
    default=True,
    prompt=True,
)
def resynthesize(input, output, n_units, dp_lambda, deduped):
    """Decodes a directory of text files into a audio files"""

    input = Path(input)
    output = Path(output)

    click.echo(f"Decoding text files from {input.absolute()}")
    click.echo(f"Saving encoded audio files to {output.absolute()}")
    click.echo(f"Using {n_units} units and {dp_lambda} lambda")
    click.echo("Loading model...")

    model = GSLM(n_units=n_units, dp_lambda=dp_lambda).cuda().eval()

    for txt_path in tqdm(sorted(list(input.rglob("*.txt")))):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                unicode_text = f.read()

            wav, sr = model.decode_unicode(unicode_text, deduped=deduped)

            out_path = output / txt_path.relative_to(input).with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(out_path, wav.cpu(), sr)

        except Exception as e:
            click.echo(f"Failed to encode {txt_path}")
            (output / "errors.txt").write_text(f"{txt_path}\n{e}\n\n")


if __name__ == "__main__":
    resynthesize()
