# SF_Net: Frame Interpolation Neural Network

SF_Net is a video frame interpolation system that generates intermediate frames between two existing frames, useful for slow-motion video generation, frame rate conversion, and video enhancement. This project implements multiple neural network architectures for high-quality frame interpolation.

## Features

- **Multiple model architectures**:
  - **FilmNet**: A pyramid-based architecture using feature extraction, flow estimation, and fusion
  - **AzadeganNet**: A 3D convolution-based approach for frame interpolation
  - **BothNet**: A hybrid approach combining both architectures for improved performance

- **Advanced interpolation techniques**:
  - Multi-scale feature extraction with pyramid levels
  - Bidirectional flow estimation between frames
  - Feature warping using predicted flows
  - Sophisticated feature fusion mechanisms

- **Comprehensive training framework**:
  - Configurable training options via Gin
  - Advanced data augmentation during training
  - Learning rate scheduling with exponential decay
  - TensorBoard integration for monitoring progress

## Project Structure

The project is organized into the following directories:

```
SF_Net-v1/
├── config/            # Configuration files for different models
├── docs/              # Documentation files
├── scripts/           # Training and evaluation scripts
├── src/               # Source code
│   ├── data/          # Data loading and augmentation
│   ├── models/        # Model architectures
│   └── utils/         # Utility functions
└── setup.py           # Package installation
```

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.4 or higher
- CUDA and cuDNN for GPU acceleration (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hamidazad2001/SF_Net-v1.git
   cd SF_Net-v1
   ```

2. Install the package in development mode:
   ```bash
   python install.py
   ```

   Or manually:
   ```bash
   pip install -e .
   ```

## Usage

### Training

To train a model, use the train.py script with the appropriate configuration file:

```bash
python scripts/train.py --gin_config config/film_net-Style.gin --base_folder ./Checkpoint --label training_run
```

Options:
- `--gin_config`: Path to the Gin configuration file (options: `film_net-Style.gin`, `azadegan_net-Style.gin`, `both_net-Style.gin`)
- `--base_folder`: Path to save checkpoints and summaries
- `--label`: Descriptive label for this training run
- `--mode`: Distributed strategy approach (`cpu` or `gpu`)

### Evaluation

To evaluate a trained model, use the eval.py script:

```bash
python scripts/eval.py --model_path ./Checkpoint/training_run/saved_model
```

### TensorBoard Monitoring

To monitor training progress, use TensorBoard:

```bash
tensorboard --logdir=Checkpoint
```

## Code Organization

- **src/models/**: Model architecture components
  - `model_lib.py`: Model creation and initialization
  - `interpolator.py`: Main interpolation logic
  - `feature_extractor.py`: Feature extraction from input images
  - `pyramid_flow_estimator.py`: Flow estimation between frames
  - `fusion.py`: Feature fusion to generate output frames

- **src/data/**: Data handling
  - `data_lib.py`: Dataset creation and preprocessing
  - `augmentation_lib.py`: Data augmentation for training

- **src/utils/**: Utilities
  - `options.py`: Configuration options
  - `util.py`: Utility functions
  - `losses.py`: Loss functions
  - `metrics_lib.py`: Evaluation metrics

- **scripts/**: Scripts for training and evaluation
  - `train.py`: Main training script
  - `train_lib.py`: Training utilities
  - `eval.py` and `eval_lib.py`: Evaluation scripts

## Architecture

### FilmNet Architecture

The FilmNet architecture uses a pyramid-based approach:
1. Feature extraction at multiple scales forming a feature pyramid
2. Flow estimation between frames using the feature pyramids
3. Feature warping using the predicted flows
4. Feature fusion to generate the interpolated frame

### AzadeganNet Architecture

The AzadeganNet architecture uses 3D convolutions to process the input frames as a 3D volume, directly learning the interpolation function from the spatial-temporal data.

### BothNet Architecture

BothNet combines the strengths of both FilmNet and AzadeganNet by fusing their outputs to leverage both pyramid-based flow prediction and 3D convolution approaches.

## Configuration

Model and training configurations are handled through Gin config files located in the `config/` directory:
- `film_net-Style.gin`: Configuration for FilmNet model
- `azadegan_net-Style.gin`: Configuration for AzadeganNet model
- `both_net-Style.gin`: Configuration for BothNet model

## Citation

If you use this code in your research, please cite:

```
@article{SF_Net,
  title={SF_Net: Frame Interpolation Neural Network},
  author={Hamid Azadegan},
  journal={arXiv preprint},
  year={2023}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

This project builds upon research in neural video frame interpolation, including:
- FILM: Frame Interpolation for Large Motion
- Other relevant works in the field of neural frame interpolation