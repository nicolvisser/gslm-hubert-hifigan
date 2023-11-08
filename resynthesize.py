from pathlib import Path

import click
import torchaudio
from tqdm import tqdm

from gslm import GSLM


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("-n", "--n_units", type=click.INT, default=500)
@click.option("-l", "--dp_lambda", type=click.FLOAT, default=0.0)
@click.option("-e", "--extension", default=".flac")
def resynthesize(input, output, n_units, dp_lambda, extension):
    """Resynthesize a directory of audio files."""

    input = Path(input)
    output = Path(output)

    click.echo(f"Resynthesizing audio files from {input.absolute()}")
    click.echo(f"Using {n_units} k-means units and a DPDP lambda of {dp_lambda}")
    click.echo(f"Saving resynthesized audio files to {output.absolute()}")
    click.echo(f"Using {n_units} units and {dp_lambda} lambda")
    click.echo("Loading model...")

    model = GSLM(n_units=n_units, dp_lambda=dp_lambda).cuda().eval()

    for wav_path in tqdm(sorted(list(input.rglob(f"*{extension}")))):
        try:
            wav, sr = torchaudio.load(wav_path)

            wav = wav.cuda()
            units = model.encode(wav, sr, dedupe=True)
            wav_, sr = model.decode(units, deduped=True)

            out_path = output / wav_path.relative_to(input).with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(out_path, wav_.cpu(), sr)
        except Exception as e:
            click.echo(f"Failed to resynthesize {wav_path}")
            (output / "errors.txt").write_text(f"{wav_path}\n{e}\n\n")


if __name__ == "__main__":
    resynthesize()
