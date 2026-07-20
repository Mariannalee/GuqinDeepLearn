# Guqin Jianzipu Trainer

This project trains a supervised image model for guqin reduced-character notation
(減字譜), then uses a mode/string/hui/technique mapping table to output jianpu
(簡譜).

## Setup

```bash
python3 -m pip install torch pillow numpy
python3 guqin.py init
```

## Files

- `data/images/`: put training images here.
- `data/labels.csv`: one row per labeled image.
- `data/mapping.csv`: your music-theory lookup table from guqin components to
  jianpu and pitch.
- `models/guqin_jianzipu.pt`: trained model output.

`data/labels.csv` columns:

```csv
image_path,mode,string,hui,technique,jianpu
data/images/example.png,黃鐘正調,四,九,勾,6
```

`data/mapping.csv` columns:

```csv
mode,string,hui,technique,jianpu,pitch
黃鐘正調,四,九,勾,6,A/la
黃鐘正調,五,七,,6,A/la
```

## Train

```bash
python3 guqin.py train --labels data/labels.csv --mapping data/mapping.csv
```

The model learns five supervised labels from each image:

- `mode`: 調式, such as `黃鐘正調`
- `string`: 弦, such as `四`
- `hui`: 徽位, such as `九`
- `technique`: 指法, such as `勾`
- `jianpu`: 簡譜, such as `6`

## Predict

```bash
python3 guqin.py predict data/images/new_symbol.png
python3 guqin.py predict data/unlabeled_folder
```

Prediction prints CSV to the terminal:

```csv
image_path,mode,string,hui,technique,jianpu,pitch
...
```

## Supervised Review

Use `review` to inspect predictions and append accepted or corrected labels:

```bash
python3 guqin.py review data/unlabeled_folder
python3 guqin.py train
```

That is the intended machine-learning loop:

1. Label a small starter set in `data/labels.csv`.
2. Train.
3. Predict unknown images.
4. Accept or correct predictions with `review`.
5. Train again with the larger corrected dataset.

