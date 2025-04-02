"""
This script provides an example of using a YOLO model for object detection inference.

Model weights: https://huggingface.co/rgautroncgiar/croppie_coffee_ug

Components:
1. CLASS_NAMES: A dictionary mapping class indices to human-readable class names.
2. model_fn: Loads the YOLO model from the specified directory.
3. input_fn: Converts a PIL.Image into an input suitable for YOLO inference.
4. predict_fn: Runs inference on the input and returns YOLO Results objects.
5. output_fn: Processes YOLO results to extract class names, confidences, and bounding boxes.

Note:
- The CLASS_NAMES should match the classes used during the model's training.
- The model_fn must take the full model path as an argument and return a loaded model object.
"""

from typing import Tuple, List
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Complete and total class mapping defined at the top
CLASS_NAMES = {
    0: "dark_brown_cherry",
    1: "green_cherry",
    2: "red_cherry",
    3: "yellow_cherry"
}

def model_fn(model_path: str, device: str = "cpu") -> YOLO:
    model = YOLO(model_path)
    model.to(device)
    return model

def input_fn(image: Image.Image, device: str = "cpu") -> np.ndarray:
    """
    Convert a PIL.Image into a NumPy array ready for YOLO inference.
    
    Args:
        image: PIL.Image object
        device: Device to place the processed input on ("cpu" or "cuda")
        
    Returns:
        NumPy array of the image in RGB format
    """
    # For YOLO models, we keep the input as a NumPy array
    # The YOLO model itself will handle moving data to the right device
    # during inference based on where the model is loaded
    return np.array(image.convert("RGB"))

def predict_fn(image_array: np.ndarray, model: YOLO) -> list:
    """
    Run inference and return YOLO Results objects.
    """
    return model(image_array)

def output_fn(
    predictions: list,
    image: Image.Image
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Convert YOLO results into nested arrays of (class_names, confidences, bounding_boxes).
    
    Args:
        predictions: List of YOLO Results objects
        image: Original PIL Image
        
    Returns:
        Tuple of 3 lists:
        - class_names: List[np.ndarray[str]] - List of numpy arrays of class name strings,
          one array per image.
        - confidences: List[np.ndarray[float]] - List of numpy arrays of confidence scores,
          one array per image.
        - bounding_boxes: List[np.ndarray[float]] with shape (num_boxes, 4) containing [xmin, ymin, xmax, ymax]
          for each image.
    """
    # Initialize lists to store results for each prediction/image
    all_classes = []
    all_confidences = []
    all_boxes = []

    for result in predictions:
        # For each prediction, create separate lists
        image_classes = []
        image_confidences = []
        image_boxes = []
        
        if hasattr(result.boxes, "data"):
            boxes = result.boxes.data.cpu().numpy()
            for x1, y1, x2, y2, conf, cls in boxes.tolist():
                image_classes.append(CLASS_NAMES.get(int(cls), "unknown"))
                image_confidences.append(conf)
                image_boxes.append([x1, y1, x2, y2])
        
        # Convert to numpy arrays and append to the main lists
        all_classes.append(np.array(image_classes, dtype=str))
        all_confidences.append(np.array(image_confidences, dtype=float))
        all_boxes.append(np.array(image_boxes, dtype=float))

    return all_classes, all_confidences, all_boxes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    my_model_path = "examples/yolo_v8_cherry/model.pt"
    model = model_fn(my_model_path, device)

    # Process an image
    img = Image.new("RGB", (224, 224), (255, 255, 255))
    model_input = input_fn(img, device)

    # Run inference
    preds = predict_fn(model_input, model)

    # Process the results
    classes, confidences, boxes = output_fn(preds, img)

    print("Classes:", classes)
    print("Confidences:", confidences)
    print("Boxes:", boxes)
