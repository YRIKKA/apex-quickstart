"""
This script provides an example of using a YOLO model for object detection inference.

Components:
1. CLASS_MAPPING: A dictionary mapping class indices to human-readable class names.
2. model_fn: Loads the YOLO model from the specified directory.
3. input_fn: Converts a PIL.Image into an input suitable for YOLO inference.
4. predict_fn: Runs inference on the input and returns YOLO Results objects.
5. output_fn: Processes YOLO results to extract class names, confidences, and bounding boxes.

Note:
- The CLASS_MAPPING should match the classes used during the model's training.
- The model_fn must take the full model path as an argument and return a loaded model object.
"""

from typing import Tuple
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Complete and total class mapping defined at the top
CLASS_MAPPING = {
    0: "dark_brown_cherry",
    1: "green_cherry",
    2: "red_cherry",
    3: "yellow_cherry"
}


def model_fn(model_dir: str) -> YOLO:
    return YOLO(model_dir)

def input_fn(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL.Image into a NumPy array ready for YOLO inference.
    """
    return np.array(image.convert("RGB"))

def predict_fn(image_array: np.ndarray, model: YOLO) -> list:
    """
    Run inference and return YOLO Results objects.
    """
    return model(image_array)

def output_fn(
    predictions: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert YOLO results into (class_names, confidences, bounding_boxes).
    - class_names: np.ndarray[str]
    - confidences: np.ndarray[float]
    - bounding_boxes: np.ndarray[float] shape=(N,4) [xmin, ymin, xmax, ymax]
    """
    class_list, conf_list, box_list = [], [], []

    for result in predictions:
        if hasattr(result.boxes, "data"):
            boxes = result.boxes.data.cpu().numpy()
            for x1, y1, x2, y2, conf, cls in boxes.tolist():
                class_list.append(CLASS_MAPPING.get(int(cls), "unknown"))
                conf_list.append(conf)
                box_list.append([x1, y1, x2, y2])

    return (
        np.array(class_list, dtype=str),
        np.array(conf_list, dtype=float),
        np.array(box_list, dtype=float)
    )


if __name__ == "__main__":
    # Load the model
    my_model_path = "yolov8_cherry_detection_example/model.pt"
    model = model_fn(my_model_path)

    # Process an image
    img = Image.new("RGB", (224, 224), (255, 255, 255))
    model_input = input_fn(img)

    # Run inference
    preds = predict_fn(model_input, model)

    # Process the results
    classes, confidences, boxes = output_fn(preds)

    print("Classes:", classes)
    print("Confidences:", confidences)
    print("Boxes:", boxes)
