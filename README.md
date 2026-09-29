# Fashion-MNIST CNN Hyperparameter Tuning with Optuna

A PyTorch notebook that uses [Optuna](https://optuna.org/) to automatically search for the best convolutional neural network (CNN) architecture and training settings for classifying Fashion-MNIST clothing images.

## Overview

Instead of hand-tuning a CNN, this project builds a **dynamic CNN** whose depth, width, and training configuration are all driven by hyperparameters, then lets Optuna run many trials to find the combination that gives the highest accuracy.

**Best result (from the recorded run):** **92.23% accuracy** on the held-out split (trial 27).

## Dataset

- **Fashion-MNIST** (`fashion-mnist_train.csv`), 60,000 grayscale 28×28 images, 10 clothing classes
- CSV format: first column is `label`, remaining 784 columns are pixel values (`pixel1`…`pixel784`)
- Download from [Kaggle – Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- Split: 80% train / 20% test (`random_state=42`)

## Pipeline

1. **Load & explore**: read the CSV, preview a 4×4 grid of sample images
2. **Preprocess**: scale pixels to [0, 1]
3. **Augmentation (train only)**: random rotation (±10°), horizontal flip, random translation (up to 10%)
4. **Model**: `DynamicCNN`, each conv block is `Conv2d → ReLU → BatchNorm → MaxPool`, followed by configurable fully connected layers with dropout
5. **Tuning**: Optuna study (`direction='maximize'`, `MedianPruner`) running 50 trials, optimizing accuracy

## Search Space

| Hyperparameter | Range / Choices |
|---|---|
| Conv layers | 1 – 3 |
| Filters per conv layer | 16, 32, 64, 128 |
| Kernel size | 3, 5 |
| FC layers | 1 – 3 |
| FC layer size | 64, 128, 256 |
| Dropout | 0.2 – 0.5 |
| Weight decay | 1e-5 – 1e-2 (log) |
| Learning rate | 1e-4 – 1e-2 (log) |
| Optimizer | SGD, Adam, RMSprop |
| Batch size | 32, 64, 128 |
| Epochs | 10 – 30 |

## Best Hyperparameters Found

| Parameter | Value |
|---|---|
| Conv layers | 3 |
| Filters | 64 |
| Kernel size | 5 |
| FC layers | 1 |
| FC layer size | 256 |
| Dropout | 0.290 |
| Weight decay | 2.2e-5 |
| Learning rate | 2.6e-4 |
| Optimizer | Adam |
| Batch size | 128 |
| Epochs | 15 |
| **Accuracy** | **0.9223** |

Adam with a 3-layer, 64-filter, 5×5 CNN dominated the top trials.

## Getting Started

### Run in Google Colab
1. Open `fashion_mnist_cnn_optuna.ipynb` in Colab
2. Set the runtime to **GPU** (Runtime → Change runtime type)
3. Upload `fashion-mnist_train.csv` to `/content/`
4. Run all cells

### Run locally
```bash
pip install optuna torch torchvision pandas scikit-learn matplotlib
jupyter notebook fashion_mnist_cnn_optuna.ipynb
```
Update the CSV path in the data-loading cell if you're not using `/content/`.

## Notes & Possible Improvements

- **Metric is measured on the same split used for selection.** Tuning on the test set can give optimistic results; use a separate validation split and keep a final untouched test set.
- **Pruning isn't active.** `MedianPruner` is set, but the objective never calls `trial.report()` / `trial.should_prune()`, so no trials are pruned. Reporting accuracy per epoch would speed up the search.
- **Deprecated Optuna calls.** Replace `suggest_uniform` and `suggest_loguniform` with `suggest_float(..., log=True)`.
- **Horizontal flip is fine for clothing** but may not suit every dataset.
- **Use the official test set** (`fashion-mnist_test.csv`) for a final unbiased evaluation.
- Retrain the best configuration and save the model weights (`torch.save`).

## Tech Stack

Python · PyTorch · torchvision · Optuna · scikit-learn · pandas · matplotlib

## License

MIT (or your preferred license)
