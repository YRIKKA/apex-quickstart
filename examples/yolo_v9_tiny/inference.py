from typing import Tuple

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

def model_fn(weights_file: str, device: str = "cpu"):
    """
    Load a YOLO model from weights file and move it to specified device.

    Args:
        weights_file (str): Path to the YOLO model weights file
        device (str, optional): Device to load model to. Defaults to "cpu"

    Returns:
        YOLO: Loaded YOLO model instance
    """
    model = YOLO(weights_file)
    model.to(device)
    return model

def _letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Helper function to resize and pad an image while maintaining aspect ratio.

    Args:
        image (PIL.Image): Input image to resize and pad
        new_shape (tuple or int, optional): Target size. Defaults to (640, 640)
        color (tuple, optional): Padding color (RGB). Defaults to (114, 114, 114)

    Returns:
        PIL.Image: Resized and padded image
    """
    # Convert to RGB just in case
    image = image.convert('RGB')
    shape = image.size  # (width, height)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute new image dimensions
    new_unpad = (int(round(shape[0] * r)), int(round(shape[1] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2
    dh /= 2

    # Resize image
    image = image.resize(new_unpad, Image.Resampling.BILINEAR)

    # Create a new image with padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Pad with color
    padded = Image.new('RGB', (new_shape[1], new_shape[0]), color)
    padded.paste(image, (left, top))
    return padded

def input_fn(
        image: Image.Image, 
        device: str = "cpu",
        target_size: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
    """
    Preprocess a PIL Image for YOLO inference by resizing, padding, and converting to tensor.

    Args:
        image (PIL.Image): Input image to preprocess
        device (str, optional): Device to place tensor on. Defaults to "cpu"
        target_size (Tuple[int, int], optional): Target size for letterboxing. Defaults to (640, 640)

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 3, H, W), normalized to [0, 1]

    Raises:
        ValueError: If input is not a PIL.Image
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL image.")
    
    # Letterbox image to a shape divisible by 32 (assumed to be (640, 640))
    image_letterboxed = _letterbox(image, target_size)

    # Convert to numpy array
    img_array = np.array(image_letterboxed, dtype=np.float32)
    # Normalize to 0-1
    img_array /= 255.0
    # Transpose HWC to CHW
    img_array = img_array.transpose(2, 0, 1)

    # Create a batch dimension
    pixel_values = torch.from_numpy(img_array).unsqueeze(0)

    return pixel_values

def predict_fn(image_array: np.ndarray, model: YOLO) -> list:
    """
    Run YOLO model inference on preprocessed image.

    Args:
        image_array (np.ndarray): Preprocessed image tensor from input_fn
        model (YOLO): Loaded YOLO model instance

    Returns:
        list: List of YOLO Results objects containing detections
    """
    return model(image_array)

def output_fn(
    predictions: list,
    image: Image.Image,
    target_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess YOLO predictions by converting to original image coordinates.

    Args:
        predictions (list): Raw predictions from YOLO model
        image (PIL.Image): Original input image
        target_size (Tuple[int, int], optional): Target size used for letterboxing. Defaults to (640, 640)

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: Tuple containing:
            - List where each element is an array of class names for each batch item
            - List where each element is an array of confidence scores for each batch item
            - List where each element is an array of bounding boxes (shape Nx4) in (x1, y1, x2, y2) 
              format in original image coordinates for each batch item
    """
    target_size = (640, 640)
    original_w, original_h = image.size

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    r = min(target_size[0] / original_h, target_size[1] / original_w)
    new_unpad = (int(round(original_w * r)), int(round(original_h * r)))
    dw, dh = target_size[1] - new_unpad[0], target_size[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    cls = []
    conf = []
    boxes = []

    for result_all in predictions:
        result = result_all.boxes
        n = result.cls.shape[0]
        cls_, conf_, box_ = [], [], []

        for i in range(n):
            class_id = int(result.cls[i].cpu().item())
            if class_id in CLASS_NAMES:
                cls_.append(CLASS_NAMES[class_id])
                conf_.append(float(result.conf[i].cpu().item()))
                box_.append(result.xyxy[i].cpu().numpy())

        if len(box_) > 0:
            box_ = np.array(box_)
            # Undo letterboxing
            box_[:, [0, 2]] -= dw  # x padding
            box_[:, [1, 3]] -= dh  # y padding
            box_ /= r

            # Clamp to original image boundaries
            box_[:, [0, 2]] = np.clip(box_[:, [0, 2]], 0, original_w)
            box_[:, [1, 3]] = np.clip(box_[:, [1, 3]], 0, original_h)
        else:
            box_ = np.array(box_)

        cls.append(np.array(cls_))
        conf.append(np.array(conf_))
        boxes.append(box_)

    return cls, conf, boxes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    my_model_path = "yolo_v9_tiny/yolov9t.pt"
    model = model_fn(my_model_path, device)

    # Process an example image
    img = Image.new("RGB", (640, 640), (255, 255, 255))
    model_input = input_fn(img, device)

    # Run inference
    preds = predict_fn(model_input, model)

    # Process the results
    classes, confidences, boxes = output_fn(preds, img)

    print("Classes:", classes)
    print("Confidences:", confidences)
    print("Boxes:", boxes)
