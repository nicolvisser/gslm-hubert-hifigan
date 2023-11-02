from pathlib import Path

import click
import torchaudio
from tqdm import tqdm

from gslm import GSLM


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("-n", "--n_units", default="500")
@click.option("-l", "--dp_lambda", default="0")
@click.option("-e", "--extension", default=".flac")
def resynthesize(input, output, n_units, dp_lambda, extension):
    """Encodes a directory of audio files into a unicode text file of chinese characters within region 4e00 and 9fff."""

    input = Path(input)
    output = Path(output)

    click.echo(f"Encoding audio files from {input.absolute()}")
    click.echo(f"Saving encoded text files to {output.absolute()}")
    click.echo(f"Using {n_units} units and {dp_lambda} lambda")
    click.echo("Loading model...")

    model = GSLM(n_units=50, dp_lambda=0).cuda().eval()

    for wav_path in tqdm(sorted(list(input.rglob(f"*{extension}")))):
        try:
            wav, sr = torchaudio.load(wav_path)

            wav = wav.cuda()
            units = model.encode(wav, sr, dedupe=True).cpu().numpy()

            unicode_chars = [chr(u + 0x4E00) for u in units]

            unicode_text = "".join(unicode_chars)

            out_path = output / wav_path.relative_to(input).with_suffix(".txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(unicode_text)

        except Exception as e:
            click.echo(f"Failed to encode {wav_path}")
            (output / "errors.txt").write_text(f"{wav_path}\n{e}\n\n")


if __name__ == "__main__":
    resynthesize()
