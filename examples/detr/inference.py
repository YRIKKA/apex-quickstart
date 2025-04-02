"""
This script provides an example of using a DETR model for object detection inference from facebook/detr-resnet-50.

Model weights: https://huggingface.co/facebook/detr-resnet-50
"""

from typing import Tuple, List
from PIL import Image
import torch
import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig
from safetensors.torch import load_file

# Complete and total class mapping defined at the top
CLASS_NAMES = {
    0: 'N/A', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 12: 'N/A', 13: 'stop sign',
    14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog',
    19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 26: 'N/A', 27: 'backpack', 28: 'umbrella',
    29: 'N/A', 30: 'N/A', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 45: 'N/A', 46: 'wine glass', 47: 'cup',
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 66: 'N/A', 67: 'dining table', 68: 'N/A',
    69: 'N/A', 70: 'toilet', 71: 'N/A', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'N/A', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}

def model_fn(weights_file: str, device: str = "cpu") -> DetrForObjectDetection:
    """
    Loads the DETR model and processor using a local safetensors weights file.
    """
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection(config)
    state_dict = load_file(weights_file)
    model.load_state_dict(state_dict)
    model.to(device)

    return model

def input_fn(image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """
    Preprocesses a PIL image using the processor and returns pixel values tensor.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL image.")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(device)

def predict_fn(pixel_values: torch.Tensor, model: DetrForObjectDetection) -> dict:
    """
    Runs inference on the image tensor.
    """
    with torch.no_grad():
        outputs = model(pixel_values)
    return outputs

def output_fn(predictions: dict, image: Image.Image) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Post-processes the DETR outputs and returns class names, confidence scores, and pixel-level boxes.
    """
    cls = []
    conf = []
    boxes = []
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50",
                                                   revision="no_timm")
    processed_results = processor.post_process_object_detection(predictions, threshold=0.5)[0]

    cls_, conf_, box_ = [], [], []

    n = processed_results['scores'].shape[0]
    for i in range(n):
        cls_.append(CLASS_NAMES[processed_results['labels'][i].item()])
        conf_.append(processed_results['scores'][i].item())
        box_.append(processed_results['boxes'][i].detach().cpu().numpy())
    # Convert lists to numpy arrays
    cls.append(np.array(cls_))
    conf.append(np.array(conf_))
    boxes.append(np.array(box_))
    
    # Convert normalized boxes to pixel coordinates
    image_width, image_height = image.size

    converted_boxes = []
    for box_set in boxes:
        pixel_boxes = []
        for box in box_set:
            ymin = max(0, box[1] * image_height)
            xmin = max(0, box[0] * image_width)
            ymax = min(image_height, box[3] * image_height)
            xmax = min(image_width, box[2] * image_width)
            pixel_boxes.append([xmin, ymin, xmax, ymax])
        converted_boxes.append(np.array(pixel_boxes))
    
    return cls, conf, converted_boxes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a test image
    img = Image.new("RGB", (800, 600), (255, 255, 255))
    
    # Load model
    model_path = "examples/detr/model.safetensors"
    model = model_fn(model_path, device)
    
    # Process image
    image_array = input_fn(img, device)
    
    # Run inference
    results = predict_fn(image_array, model)
    
    # Get final outputs
    classes, confidences, boxes = output_fn(results, img)

    print(f"Classes: {classes}, Confidences: {confidences}, Boxes: {boxes}")
