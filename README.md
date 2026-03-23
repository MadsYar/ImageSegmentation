# ImageSegmentation with CNN
This project focuses on semantic image segmentation using PyTorch. The main goal is to compare a simple encoder-decoder network and a UNet architecture on two medical imaging datasets, and then explore how the models behave under weak supervision. The codebase includes dataset loaders, model definitions, training scripts, hyperparameter search scripts, loss functions, evaluation metrics, visualization utilities, and a notebook for experimentation.

## Project overview

The project trains segmentation models on two datasets:

- **DRIVE** for retinal vessel segmentation
- **PH2** for skin lesion segmentation

Both datasets are resized to **256 x 256** before training. The repository contains two segmentation backbones:

- **EncDec**: a simpler encoder-decoder model built with repeated convolution, pooling, bottleneck, and upsampling layers
- **UNet**: a deeper encoder-decoder model with skip connections and batch normalization

The models are trained with pixel-wise binary segmentation targets and evaluated using segmentation and classification-style metrics.

## Main components

### Dataset loading
`Dataloader.py` defines custom PyTorch dataset classes for both DRIVE and PH2.

- `DRIVE(transform)` loads retinal images and manual vessel masks from the DRIVE training folder
- `PH2(train, transform, test_split=0.2, seed=42)` loads dermoscopic images and lesion masks from PH2 and creates a train/test split using `train_test_split`

The code currently expects the datasets to exist at hardcoded local paths:

- `/dtu/datasets1/02516/DRIVE/`
- `/dtu/datasets1/02516/PH2_Dataset_images`

You will likely need to update these paths before running the code on a different machine.

### Model architectures

#### `EncDec.py`
Defines a simple encoder-decoder segmentation network. The encoder repeatedly downsamples the image using convolution and max pooling. The decoder upsamples back to the original resolution and predicts a single-channel segmentation mask.

#### `UNet.py`
Defines a UNet architecture with:

- contracting path
- expanding path
- skip connections
- batch normalization
- ReLU activations

This model is designed to preserve more spatial detail than the simpler encoder-decoder baseline.

### Training scripts
The repository contains separate training scripts for each dataset and architecture combination:

- `Train_DRIVE_enc_dec.py`
- `Train_DRIVE_UNet.py`
- `Train_PH2_enc_dec.py`
- `Train_PH2_UNet.py`

These scripts:

- detect whether CUDA is available
- create train, validation, and test splits
- build dataloaders
- define hyperparameters
- train the chosen model
- log losses and accuracies with TensorBoard
- save model weights based on final test accuracy

The DRIVE scripts use a **70 / 20 / 10** split through `random_split`, while the PH2 scripts use the custom `PH2(train=True/False)` split and then split the training portion into training and validation sets.

### Hyperparameter search
The repository also includes dedicated hyperparameter search scripts:

- `HyperParameterSearch_DRIVE_enc_dec.py`
- `HyperParameterSearch_DRIVE_UNet.py`
- `HyperParameterSearch_PH2_enc_dec.py`
- `HyperParameterSearch_PH2_UNet.py`

These scripts test combinations of:

- batch size
- step size
- learning rate
- scheduler gamma
- loss function

The search uses full grid combinations and stores TensorBoard logs in `HPSearch/`. The loss candidates in the search scripts are mainly **BCE loss** and **Dice loss**.

### Loss functions
`Loss.py` contains several loss functions for binary segmentation:

- `bce_loss`
- `dice_loss`
- `cross_entropy_loss`
- `cross_entropy_weighted_loss`
- `focal_loss`

In the provided training scripts, **BCE loss** is the default training loss.

### Evaluation metrics
`EvalMetrics.py` provides functions for evaluating segmentation predictions:

- Dice overlap
- Intersection over Union
- Accuracy
- Sensitivity
- Specificity

These are useful for comparing models beyond raw pixel accuracy.

### Data visualization and weak supervision
Two utility scripts are included for inspecting the datasets:

- `DisplayData.py` visualizes samples from DRIVE and PH2 and saves example figures
- `Annotate.py` generates weakly annotated PH2 images by placing positive and negative dots on lesion and background regions

This weak annotation script shows how sparse supervision can be added on top of the original masks and supports the project’s weak supervision experiments.

### Notebook
`model-test.ipynb` contains exploratory experiments, including:

- checking GPU availability
- visualizing both datasets
- defining and testing the encoder-decoder model in notebook form
- setting hyperparameters
- combining datasets with `ConcatDataset`
- running hyperparameter search experiments

This notebook appears to be an experimental workspace for trying out model ideas before or alongside the standalone scripts.

## File structure

```text
.
├── Annotate.py
├── Dataloader.py
├── DisplayData.py
├── EncDec.py
├── EvalMetrics.py
├── HyperParameterSearch_DRIVE_UNet.py
├── HyperParameterSearch_DRIVE_enc_dec.py
├── HyperParameterSearch_PH2_UNet.py
├── HyperParameterSearch_PH2_enc_dec.py
├── Loss.py
├── README.md
├── Train_DRIVE_UNet.py
├── Train_DRIVE_enc_dec.py
├── Train_PH2_UNet.py
├── Train_PH2_enc_dec.py
├── UNet.py
└── model-test.ipynb
```

## How to run

### 1. Install dependencies
A minimal environment should include:

```bash
pip install torch torchvision torchaudio
pip install tqdm pillow scikit-learn matplotlib tensorboard torchsummary
```

If you want to use the notebook:

```bash
pip install notebook
```

### 2. Update dataset paths
Open `Dataloader.py` and change the hardcoded dataset paths so they match your local setup.

### 3. Train a model
Example:

```bash
python Train_DRIVE_UNet.py
```

or

```bash
python Train_PH2_enc_dec.py
```

### 4. Run hyperparameter search
Example:

```bash
python HyperParameterSearch_PH2_UNet.py
```

### 5. Launch TensorBoard
```bash
tensorboard --logdir HPSearch
```

or for standard training runs:

```bash
tensorboard --logdir Results
```

## Notes and limitations

- The dataset paths are hardcoded and need to be adapted for other environments.
- The code is mainly structured around binary segmentation with one output channel.
- Some scripts contain repeated training logic that could be refactored into shared utilities.
- The repository includes both finalized scripts and exploratory notebook-based work.
- Some older helper files use `transforms.ToTensor()` while the training scripts use `torchvision.transforms.v2` with `ToImage()` and `ToDtype()`.

## Summary

Overall, this repository implements a full segmentation workflow in PyTorch: loading medical image datasets, defining encoder-decoder and UNet models, training and validating them, comparing different hyperparameter settings, visualizing the data, and experimenting with weak supervision. It is a solid project for studying how architecture choice, loss function, and annotation quality affect segmentation performance.
