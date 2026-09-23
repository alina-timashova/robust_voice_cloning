# Compression and age robust voice cloning detection

This project check whether a light CNN (LCNN), trained to distinguish fake clips from real ones produced by adults, generalise to children's voices and Opus-compressed audio versions.

## Files

| File | What it does |
|---|---|
| `data_prep.py` | Preprocesses adult speech from **CodecFake** (`ajaykarthick/codecfake-audio`). It keeps real clips (`R`) plus SpeechTokenizer (`F02`) and EnCodec (`F04`) fakes and splits them by speaker (70/20/10). Each clip is saved in a *clean* version (16 kHz mono) and a *compressed* version (Opus, 16 kbps). Runs in parallel with `multiprocessing`. It also provides `load_split()`, which the model script uses. |
| `children_data_prep.py` | Preprocesses child speech from **Samromur Children** (`language-and-voice-lab/samromur_children`). This dataset has only real clips, so it makes the fakes itself by running each clip through SpeechTokenizer (`F02`) and EnCodec (`F04`) in batches on the GPU. It randomly picks as many child speakers as there are adult speakers. The output format is the same as `data_prep.py`. |
| `test_set_DAC.py` | Builds a test-only set with an **unseen codec (DAC, 16 kHz)**. It takes real test-split clips and adds a DAC-reconstructed copy of each. This is used to check how well the model generalizes to a codec it never saw in training. |
| `model_melCNN_experiments.py` | Converts audio to log-mel spectrograms (80 mels × 126 frames), then trains and tests the LCNN for the chosen experiment. It reports accuracy, precision, recall, F1, the confusion matrix and EER. |

## How it runs

1. `python data_prep.py`: adult data → `~/nn-dataprep/codecfake_preprocessed`
2. `python children_data_prep.py`: child data → `~/nn-dataprep/samromur_preprocessed_sp` (must run after step 1, because it counts the adult speakers)
3. `python test_set_DAC.py`: DAC test set
4. `python model_melCNN_experiments.py`: set `ACTIVE_EXPERIMENT`, `TEST_ONLY` and `FINAL_TEST_RUN` at the top of the file first
