# Bangla-Handwritten-OCR

This project implements an Optical Character Recognition (OCR) system specifically designed for handwritten Bengali text. It leverages a hybrid deep learning architecture, combining a Convolutional Neural Network (CNN) for feature extraction with a Transformer Decoder for sequence generation.

## Architecture

The core model (`BanglaOCR` in `model.py`) is inspired by DETR (DEtection TRansformer) and adapted for OCR:
- **Encoder**: A ResNet-18 model (from `torchvision`) is used to extract visual features from the input images.
- **Positional Encoding**: Both 1D sine/cosine positional encodings (for the sequence) and 2D spatial embeddings (for the image features) are used to retain spatial awareness.
- **Decoder**: A PyTorch `TransformerDecoder` takes the CNN features and auto-regressively predicts the sequence of characters.

## Project Structure

- `main.py`: The main entry point script for running both training and batch inference.
- `model.py`: Defines the `BanglaOCR` PyTorch model, including the ResNet encoder and Transformer decoder.
- `train_model.py`: Contains the training loop, validation logic, loss calculation (with Label Smoothing), and model saving.
- `inference.py`: Runs batch inference on validation data and calculates metrics like Character Error Rate (CER), Word Error Rate (WER), and Sequence Error Rate (SER).
- `single_image_inference.py`: A standalone script to run inference on a single image. Includes necessary image preprocessing (binarization, resizing).
- `services.py`: Includes helper classes such as `Tokenizer`, `DataGenerator` (Custom Dataset), `LabelSmoothing`, and `EarlyStopper`.
- `utils.py`: Utility functions for calculating maximum sequence lengths, loading files, and generating training plots.
- `train_config.yaml`: Configuration parameters for training (epochs, batch size, patience, model hyper-parameters).
- `inference_config.yaml`: Configuration for inference (model name and specific image for single image inference).

## Dataset Directory Structure

To train and validate the model, ensure your dataset is organized in the root directory as follows:

```text
data/
├── Train/
│   ├── Images/       # Training images
│   └── Annotations/  # Corresponding ground truth text files
└── Validation/
    ├── Images/       # Validation images
    └── Annotations/  # Corresponding ground truth text files
```

Additionally, the project requires a `charset/` directory with printable characters:
- `charset/printable1.txt`
- `charset/printable2.txt`

## Setup & Installation

Install the required dependencies. It is recommended to use a virtual environment:

```bash
pip install torch torchvision numpy opencv-python editdistance omegaconf matplotlib mlconfig
```

*Note: Ensure you install the version of PyTorch compatible with your CUDA setup.*

## Usage

### 1. Training

To train the model, first configure the hyper-parameters in `train_config.yaml`:

```yaml
epoch: 20
patience: 50
batch_size: 4
model:
  d_model: 256  # Transformer hidden dimension
  num_decoder_layers: 4  # Number of decoder layers
  nheads: 4  # Number of attention heads
```

Then, run the main script in `train` mode:

```bash
python main.py train
```
* The trained model checkpoints will be saved in the `Saved Checkpoints/` directory.
* Training logs and loss plots will be saved in the `Outputs/` directory.
* A `tokenizer.pk` file will also be generated and saved in `Outputs/`.

### 2. Batch Inference & Evaluation

To evaluate the model on the validation dataset and calculate CER, WER, and SER:

1. Open `inference_config.yaml` and set `model_name` to your saved model's name (without the `.pth` extension).
2. Run the main script in `inference` mode:

```bash
python main.py inference
```
* Inference logs and evaluations will be saved in `Outputs/`.

### 3. Single Image Inference

To predict the text of a single handwritten image:

1. Place the image inside the `data/` directory.
2. Update `inference_config.yaml` with the `model_name` and the `image` file name.
3. Run the script:

```bash
python single_image_inference.py
```

The script automatically applies preprocessing (Gaussian Blur, Adaptive Thresholding, and resizing) before passing the image through the Transformer model.

## Outputs

- **Saved Checkpoints/**: Contains `.pth` model weights.
- **Outputs/**: Contains `tokenizer.pk`, `train_logs.txt`, `inference_logs.txt`, and generated training loss plots.