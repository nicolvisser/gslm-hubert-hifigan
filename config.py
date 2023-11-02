class ExperimentConfig:
    librispeech_dir = "/home/nicolvisser/datasets/LibriSpeech"
    output_dir = "/home/nicolvisser/workspace/gslm-hubert-hifigan/output"
    dp_lmbdas = {
        50: [0],
        100: [0, 4, 8, 12, 16, 20],
        200: [0, 4, 8, 12, 16, 20],
        500: [0, 4, 8, 12, 16, 20, 24, 28],
        1000: [0, 4, 8, 12, 16, 20],
        2000: [0, 4, 8, 12, 16, 20],
    }
