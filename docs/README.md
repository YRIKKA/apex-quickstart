# APEX Model Package Guide

This guide explains how to prepare your object detection model for evaluation with APEX.

## Model Package Structure

Your model package must include these required files:

```
my_model/
├── inference.py     # Implements required interface functions
├── model.pt        # Your model weights file
└── manifest.json   # Configuration specifying entry points
```

> 📋 **Quick Start**: Use the `examples/yolo_v9_tiny` directory as your template! This working example can be easily adapted for your own models.

## Required Files

### 1. inference.py

Your `inference.py` must implement these key functions:

#### CLASS_NAMES Dictionary
```python
CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car',
    # ... other classes ...
}
```

#### Required Functions

APEX uses these functions to interface with your model during evaluation. Each function handles a specific part of the inference pipeline:

```python
def model_fn(model_dir: str, device: str) -> Any:
    """
    Load model from disk and move to specified device.

    Args:
        model_dir: Path to directory containing model weights
        device: Either 'cpu' or 'cuda' - function must move model to this device

    Returns:
        Model object (any format your predict_fn accepts)
    """
    pass

def input_fn(
    image: PIL.Image.Image, 
    device: str,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Preprocess input image for model inference.

    Args:
        image: PIL Image object
        device: Either 'cpu' or 'cuda' - function must move data to this device
        target_size: Optional target size for letterboxing

    Returns:
        Processed input in format your model requires (e.g., numpy array)
    """
    pass

def predict_fn(image_array: np.ndarray, model: Any) -> Any:
    """
    Run model inference on preprocessed input.

    Args:
        image_array: Output from input_fn
        model: Model object returned by model_fn

    Returns:
        Raw prediction results (any format your output_fn accepts)
    """
    pass

def output_fn(
    predictions: Any,
    image: PIL.Image.Image,
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Post-process predictions to required format.

    Args:
        predictions: Output from predict_fn
        image: Original input PIL Image
        target_size: Optional target size used for letterboxing

    Returns:
        Tuple of (class_names, confidences, bounding_boxes):
            - class_names: List[np.ndarray[str]] - Arrays of class name strings
            - confidences: List[np.ndarray[float]] - Arrays of confidence scores
            - bounding_boxes: List[np.ndarray[float]] - Arrays of shape (N, 4) 
              containing [xmin, ymin, xmax, ymax] in pixels
    """
    pass
```

### 2. Model Weights File

Include your trained model weights file (e.g., `model.pt`, `weights.h5`).

### 3. manifest.json

The `manifest.json` file tells APEX where to find your files in the package. It must specify:

```json
{
    "entry_point": "inference.py",  // Name of your inference script
    "model_file": "model.pt"       // Name of your model weights file
}
```

This configuration allows you to:
- Use any name for your inference script (e.g., `predict.py`, `detect.py`)
- Use any name for your model weights (e.g., `weights.h5`, `checkpoint.pth`)

Just make sure the names in `manifest.json` match your actual filenames.

## Supported Libraries

Your `inference.py` can use these pre-installed packages:

### Deep Learning
- PyTorch (2.4+) and Torchvision (0.19+)
- TensorFlow (2.19+) and Keras (3.x)
- PyTorch Lightning (2.x)
- Model libraries: Timm, YOLOv8, EfficientNet

### Computer Vision
- OpenCV (4.8+)
- albumentations
- scikit-image
- Pillow
- pycocotools
- imageio

### Other Libraries
- Hugging Face Transformers (4.37+)
- numpy, pandas, scipy
- scikit-learn (1.3+)
- matplotlib, seaborn

> ⚡ **Note**: The environment uses CUDA 12.1 for GPU acceleration.

🙋‍♀️ **Don't see a package you need?** No problem — contact us to learn about getting your own private API environment with custom dependencies.
## Example Implementation

See complete working examples in:
- `examples/yolo_v9_tiny/` - YOLOv9 implementation
- `examples/yolo_v8_cherry/` - YOLOv8 implementation
- `examples/detr/` - DETR implementation

## Package Size Limit

Your model package tarball must be less than 4GB in size.

## Need Help?

- [API Documentation](https://yrikka.github.io/apex-quickstart/)
- Contact: help@yrikka.com 