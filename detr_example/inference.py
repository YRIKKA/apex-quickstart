"""
This script provides an example of using a DETR model for object detection inference from facebook/detr-resnet-50.
"""

from typing import Tuple
import numpy as np
import torch
from PIL import Image
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from safetensors.torch import load_file

# COCO classes mapping
CLASS_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush"
}

def model_fn(weights_file: str, device: str = "cpu") -> Tuple[DetrForObjectDetection, DetrImageProcessor]:
    """
    Load the DETR model and processor from local files.
    
    Args:
        weights_file: Path to the model weights file
        device: Device to load model to ("cpu" or "cuda")
        
    Returns:
        Tuple of (model, processor)
    """
    # Load a default configuration (here we use the facebook/detr-resnet-50 config)
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    # Instantiate the model with the configuration
    model = DetrForObjectDetection(config)
    # Load the state dict from the safetensors file
    state_dict = load_file(weights_file)
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    # Load the image processor (remains unchanged)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model.to(device)
    return model, processor

def input_fn(image: Image.Image, device: str = "cpu") -> np.ndarray:
    """
    Convert PIL image to numpy array.
    
    Args:
        image: PIL Image to process
        device: Device to place tensors on
        
    Returns:
        Numpy array of the image
    """
    return np.array(image)

def predict_fn(image_array: np.ndarray, model: Tuple[DetrForObjectDetection, DetrImageProcessor]) -> dict:
    """
    Run DETR inference using model and processor.
    
    Args:
        image_array: Numpy array from input_fn
        model: Tuple of (model, processor) from model_fn
        
    Returns:
        Dictionary containing processed predictions
    """
    detr_model, processor = model
    
    # Process image using the processor
    inputs = processor(images=image_array, return_tensors="pt")
    inputs = {k: v.to(detr_model.device) for k, v in inputs.items()}
    
    # Run inference
    outputs = detr_model(**inputs)
    
    # Post-process outputs
    target_sizes = torch.tensor([image_array.shape[:2][::-1]])
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes, 
        threshold=0.9
    )[0]
    
    return results

def output_fn(
    predictions: dict,
    image: Image.Image
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DETR results into required format.
    
    Args:
        predictions: Results from predict_fn
        image: Original input image
        
    Returns:
        Tuple of (class_names, confidences, boxes) where:
        - class_names: np.ndarray of str class names
        - confidences: np.ndarray of float confidence scores
        - boxes: np.ndarray of float boxes in [x1,y1,x2,y2] format
    """
    # Convert to numpy arrays - using detach() to remove gradient tracking
    scores = predictions["scores"].detach().cpu().numpy()
    labels = predictions["labels"].detach().cpu().numpy()
    boxes = predictions["boxes"].detach().cpu().numpy()
    
    # Convert label indices to class names
    class_names = np.array([CLASS_NAMES[label.item()] for label in labels])
    
    return class_names, scores, boxes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a test image
    img = Image.new("RGB", (800, 600), (255, 255, 255))
    
    # Load model and processor
    model_path = "detr_example/model.safetensors"
    model = model_fn(model_path, device)
    
    # Process image
    image_array = input_fn(img, device)
    
    # Run inference
    results = predict_fn(image_array, model)
    
    # Get final outputs
    classes, confidences, boxes = output_fn(results, img)
    
    print("\nDetections:")
    for cls, conf, box in zip(classes, confidences, boxes):
        print(f"Class: {cls}, Confidence: {conf:.3f}, Box: {box.tolist()}")
