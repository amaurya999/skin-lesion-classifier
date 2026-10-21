# Skin Lesion Classifier (Research Demo)

> **Research and educational demo only.** This is not a medical device and does not provide a diagnosis. Do not use it to make health decisions.

A PyTorch image classifier for seven types of pigmented skin lesions, served through a FastAPI web app with a simple upload page.

## What it does

- Fine-tunes a pretrained ResNet18 on the HAM10000 dermatoscopic dataset (10,015 images, 7 classes)
- Splits data by lesion, not by image, to prevent train/test leakage
- Handles heavy class imbalance with a class-weighted loss and macro-F1 as the selection metric
- Serves predictions through a REST API (`/predict`) and a browser upload page

## Results (validation set, 1,532 images)

| Model | Accuracy | Macro-F1 |
| --- | --- | --- |
| Frozen backbone (baseline) | 0.581 | 0.372 |
| Last block fine-tuned | 0.680 | 0.496 |

Melanoma recall is 0.68 and basal cell carcinoma recall is 0.56. The best epoch was chosen on this same validation set, so these numbers are slightly optimistic. A one-time test-set evaluation is planned.

## Project structure

- `src/config.py` - paths and settings
- `src/dataset.py` - metadata, lesion-level splits, augmentation, data loaders
- `src/model.py` - ResNet18 with a new classification head
- `src/train.py` - training loop (baseline and fine-tuning)
- `src/evaluate.py` - per-class report and confusion matrix
- `src/predict.py`, `src/api.py` - inference and the web API
- `static/index.html` - upload page

## Setup

1. Create and activate a virtual environment, then install dependencies:

```
   pip install -r requirements.txt
```

2. Download HAM10000 from the Harvard Dataverse. Put all images in `data/raw/images/` and the metadata file at `data/raw/HAM10000_metadata.csv`.

3. Train the baseline, then fine-tune:

```
   python -m src.train --epochs 5
   python -m src.train --finetune --epochs 4
```

4. Evaluate and run the app:

```
   python -m src.evaluate --split val --checkpoint best_finetuned.pt
   uvicorn src.api:app --port 8000
```

Model weights and data are not stored in this repository.

## Limitations

- Trained on dermatoscopic images; it performs poorly on ordinary phone photos.
- Rare classes have very few validation images, so their scores are noisy.
- Confidence values are not calibrated medical probabilities.
- HAM10000 is licensed for non-commercial use, so this project is for learning and demonstration only.

## Status

Local API and upload page work. Docker and AWS deployment are in progress.