from pathlib import Path

import click
import torchaudio
from tqdm import tqdm

from gslm import GSLM


@click.command()
@click.option(
    "-i",
    "--input_dir",
    help="Path to input directory containing audio.",
    type=click.Path(exists=True),
    prompt=True,
)
@click.option(
    "-o",
    "--output_dir",
    help="Path to output directory to hold text files.",
    type=click.Path(),
    prompt=True,
)
@click.option(
    "-n",
    "--n_units",
    help="The number of k-means units to use.",
    type=click.INT,
    prompt=True,
)
@click.option(
    "-l",
    "--dp_lambda",
    help="The lambda paramter of DPDP to use.",
    type=click.FLOAT,
    prompt=True,
)
@click.option(
    "-e",
    "--extension",
    help="The extension of the audio files in the input directory.",
    prompt=True,
)
@click.option(
    "-d",
    "--dedupe",
    help="Whether the units should be deduped.",
    default=True,
    prompt=True,
)
def resynthesize(input_dir, output_dir, n_units, dp_lambda, extension, dedupe):
    """Encodes a directory of audio files into a unicode text file of chinese characters within region 4e00 and 9fff."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    click.echo(f"Encoding audio files from {input_dir.absolute()}")
    click.echo(f"Saving encoded text files to {output_dir.absolute()}")
    click.echo(f"Using {n_units} units and {dp_lambda} lambda")
    click.echo("Loading model...")

    model = GSLM(n_units=n_units, dp_lambda=dp_lambda).cuda().eval()

    for wav_path in tqdm(sorted(list(input_dir.rglob(f"*{extension}")))):
        wav, sr = torchaudio.load(wav_path)

        wav = wav.cuda()

        unicode_text = model.encode_unicode(wav, sr, dedupe=dedupe).cpu().numpy()

        out_path = output_dir / wav_path.relative_to(input_dir).with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(unicode_text)


if __name__ == "__main__":
    resynthesize()
