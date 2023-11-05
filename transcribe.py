from pathlib import Path
import whisper
from tqdm import tqdm

from config import ExperimentConfig

paths = sorted(list(Path(ExperimentConfig.output_dir).rglob("*.wav")))

model = whisper.load_model("base.en", "cuda")

for path in tqdm(paths):
    out_path = path.with_suffix(".txt")

    if out_path.exists():
        continue

    result = model.transcribe(
        str(paths[0]),
    )

    transcription = result["text"]

    out_path.write_text(transcription)
