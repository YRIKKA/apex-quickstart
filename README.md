# YRIKKA APEX: Automated Context-Based Object Detection Model Evaluation

## Overview

This repository provides sample code and examples for using YRIKKA's APEX API - an automated context-based evaluation system for object detection models. APEX allows you to test your models in specific scenarios by describing the context in natural language.

![APEX Workflow](assets/apex-workflow.png)

**[🔑 Sign up for an APEX API key](https://apex.yrikka.com/login?client_id=3fn9ks2vmp3gdis9jvts464v31&response_type=code&scope=email+openid+phone&redirect_uri=https%3A%2F%2Fyrikka.com%2F)**

**[📚 View the complete API documentation](https://docs.yrikka.com/apex)**

**[🌐 Visit YRIKKA](https://yrikka.com)**

## Table of Contents

- [What is APEX?](#what-is-apex)
- [Getting Started](#getting-started)
  - [1. Get an API Key](#1-get-an-api-key)
  - [2. Prepare Your Model Package](#2-prepare-your-model-package)
- [Model Package Structure](#model-package-structure)
  - [Required Files](#required-files)
  - [Supported Libraries and Dependencies](#supported-libraries-and-dependencies)
  - [Example Implementation](#example-implementation)
  - [inference.py Requirements](#inferencepy-requirements)
  - [manifest.json Format](#manifestjson-format)
- [Creating Effective Context Descriptions](#creating-effective-context-descriptions)
- [Using the API](#using-the-api)
  - [Step 1: Package Your Model](#step-1-package-your-model)
  - [Step 2: Get a Presigned URL](#step-2-get-a-presigned-url)
  - [Step 3: Upload Your Model](#step-3-upload-your-model)
  - [Step 4: Submit an Evaluation Job](#step-4-submit-an-evaluation-job)
  - [Step 5: Check Job Status and Get Results](#step-5-check-job-status-and-get-results)
- [Example Model Output](#example-model-output)
- [Troubleshooting](#troubleshooting)
- [Need Help?](#need-help)

## What is APEX?

APEX is a powerful evaluation platform that tests your object detection models in customizable contexts using synthetically generated data. Using just three inputs from you, APEX:

- Creates synthetic test images based on your natural language context descriptions
- Evaluates your model against these test images
- Provides detailed performance metrics across different scenarios
- Helps identify strengths and weaknesses in your models

All you need to provide is:
1. **Natural language descriptions** of the test contexts (e.g., "detect cherries in varying lighting conditions")
2. Your **model weights** file
3. A simple **inference script** implementing the required functions

No need to create test datasets manually - APEX handles the entire evaluation process automatically.

## Getting Started

### 1. Get an API Key

1. [Sign up for an APEX API key](https://apex.yrikka.com/login?client_id=3fn9ks2vmp3gdis9jvts464v31&response_type=code&scope=email+openid+phone&redirect_uri=https%3A%2F%2Fyrikka.com%2F)
2. Verify your email address
3. Check your email for your API key

### 2. Prepare Your Model Package

Your model package must include:

- **inference.py**: Script implementing the required inference functions
- **model weights**: Your trained model file (e.g., `model.pt`)
- **manifest.json**: Configuration file specifying entry points

This repository includes an example model package in the `yolov8_cherry_detection_example` directory.

## Model Package Structure

### Required Files

```
yolov8_cherry_detection_example/
├── inference.py     # Implements required interface functions
├── model.pt         # Your model weights
└── manifest.json    # Configuration specifying entry points
```

> 📋 **Ready-to-Use Example**: Start with the `yolov8_cherry_detection_example` directory as your template! This working example can be easily adapted for your own models:
> 
> 1. Copy the entire `yolov8_cherry_detection_example` directory and rename it for your project
> 2. Update the `CLASS_MAPPING` dictionary in `inference.py` with your model's classes
> 3. Replace the model file with your own weights (keep the same name or update manifest.json)
> 4. Create a tarball of your directory following the instructions below
>
> This approach can be adapted for other object detection architectures with minimal changes to the inference functions.

### Supported Libraries and Dependencies

APEX supports the following libraries in your inference script:

| Library | Version |
|---------|---------|
| PyTorch | 2.0.0+ |
| TorchVision | 0.15.0+ |
| TensorFlow | 2.10.0+ |
| HuggingFace Transformers | 4.25.0+ |
| HuggingFace Hub | 0.12.0+ |
| ONNX Runtime | 1.13.1+ |
| Ultralytics (YOLO) | 8.0.0+ |
| NumPy | 1.23.0+ |
| OpenCV | 4.6.0+ |
| Pillow (PIL) | 9.3.0+ |
| SciPy | 1.9.0+ |
| Scikit-learn | 1.1.0+ |

Notes:
- Your inference script **must only use** the libraries listed above in the specified versions
- Additional dependencies are not currently supported
- If your inference script imports libraries not in this list, your evaluation job will fail
- Maximum package size must not exceed 4GB

> 🔮 **Coming Soon**: In a future update, we plan to support custom dependencies via requirements.txt files. Stay tuned!

### Example Implementation

The example implementation in this repository uses the [Croppie cherry detection model](https://huggingface.co/rgautroncgiar/croppie_coffee_ug), a YOLOv8 model fine-tuned for coffee cherry detection. This model detects four classes of coffee cherries:

```python
{
    0: "dark_brown_cherry", 
    1: "green_cherry", 
    2: "red_cherry", 
    3: "yellow_cherry"
}
```

### inference.py Requirements

Your `inference.py` must include the following components:

#### 1. `CLASS_MAPPING` Dictionary
You must define a dictionary mapping class indices to class names at the top of your script.

```python
# Define your class mapping - indices to human-readable class names
CLASS_MAPPING = {
    0: "dark_brown_cherry",
    1: "green_cherry",
    2: "red_cherry",
    3: "yellow_cherry"
}
```

#### 2. `model_fn(model_dir)`
Loads your model from the specified path.

```python
def model_fn(model_dir: str):
    """
    Load the model from disk.
    
    Args:
        model_dir: Full path to the model file
        
    Returns:
        Your loaded model object
    """
    # Example for PyTorch model
    import torch
    from my_model import MyModel
    
    # model_dir is already the full path to the model file
    model = MyModel()
    model.load_state_dict(torch.load(model_dir))
    return model
```

#### 3. `input_fn(image)`
Prepares input for your model.

```python
def input_fn(image):
    """
    Preprocess the input image for model inference.
    
    Args:
        image: PIL.Image object
        
    Returns:
        Processed input in format required by your model
    """
    # Example preprocessing for a CNN
    import numpy as np
    
    # Convert to numpy array
    img_array = np.array(image.convert("RGB"))
    
    # Resize, normalize, etc.
    img_array = img_array / 255.0
    
    return img_array
```

#### 4. `predict_fn(image_array, model)`
Runs inference using your model.

```python
def predict_fn(image_array, model):
    """
    Run model inference on the preprocessed input.
    
    Args:
        image_array: Output from input_fn
        model: Model object returned by model_fn
        
    Returns:
        Raw prediction results
    """
    # Example inference
    predictions = model(image_array)
    return predictions
```

#### 5. `output_fn(predictions)`
Formats the results in the required format.

```python
def output_fn(predictions):
    """
    Post-process model predictions to required format.
    
    Args:
        predictions: Output from predict_fn
        
    Returns:
        tuple of (class_names, confidences, bounding_boxes):
          - class_names: np.ndarray[str] - Array of class name strings
          - confidences: np.ndarray[float] - Array of confidence scores
          - bounding_boxes: np.ndarray[float] with shape (N,4) containing [xmin, ymin, xmax, ymax]
    """
    import numpy as np
    
    # Process your model's output to extract relevant information
    class_indices = predictions['class_ids']
    confidences = predictions['scores']
    boxes = predictions['boxes']  # [xmin, ymin, xmax, ymax] format
    
    # Map class indices to names using CLASS_MAPPING
    class_names = np.array([CLASS_MAPPING[idx] for idx in class_indices], dtype=str)
    
    return class_names, np.array(confidences, dtype=float), np.array(boxes, dtype=float)
```

### manifest.json Format

The `manifest.json` file specifies your entry points:

```json
{
  "entry_point": "inference.py",
  "model_file": "model.pt"
}
```

## Using the API

Once you have created your inference script and prepared your model files, you'll need to follow these steps to evaluate your model with APEX. A full example implementation is provided in the `demo.py` file in this repository. Below, we break down each step:

### Step 1: Package Your Model

> ⚠️ **IMPORTANT**: Your model package tarball must be less than 4GB in size.

Create a tarball of your model directory:

```python
import os
import tarfile

def create_tarball(directory, output_file):
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(directory, arcname=os.path.basename(directory))
    print(f"Created tarball '{output_file}' from directory '{directory}'")

# Usage
MODEL_DIR = "my_model_package"
TARBALL_PATH = "model_package.tar.gz"
create_tarball(MODEL_DIR, TARBALL_PATH)
```

### Step 2: Get a Presigned URL

Request a presigned URL to upload your model:

```python
import requests

def get_presigned_url(api_key):
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    response = requests.get("https://api.yrikka.com/v1/presigned", headers=headers)
    response.raise_for_status()
    return response.json()["upload_url"], response.json()["s3_uri"]

# Usage
API_KEY = "your_api_key_here"
upload_url, s3_uri = get_presigned_url(API_KEY)
```

### Step 3: Upload Your Model

Upload your tarball to the presigned URL:

```python
def upload_tarball(upload_url, file_path):
    with open(file_path, "rb") as f:
        response = requests.put(upload_url, data=f)
    response.raise_for_status()

# Usage
upload_tarball(upload_url, TARBALL_PATH)
```

#### Alternative: Using AWS CLI

If you prefer using command-line tools:

```bash
# Direct upload with curl
curl -X PUT -T model_package.tar.gz "PRESIGNED_URL_HERE"

# Or with AWS CLI (also using the presigned URL)
aws s3 cp model_package.tar.gz "PRESIGNED_URL_HERE"
```

### Step 4: Submit an Evaluation Job

Once you have uploaded a model, you may submit a job with your desired context using the model package URI you received from the presigned URL:

```python
def submit_job(api_key, model_package_uri, target_classes, context_description):
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    data = {
        "s3_model_package_uri": model_package_uri,
        "target_classes": target_classes,
        "context_description": context_description
    }
    response = requests.post("https://api.yrikka.com/v1/submit-job", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["job_id"]

# Usage
target_classes = ["dark_brown_cherry", "green_cherry", "red_cherry"]
context = (
            "Evaluate cherry detection in outdoor orchards with varying lighting conditions "
            "including direct sunlight, overcast, and dawn/dusk. Include scenarios with cherries "
            "partially hidden by leaves, different stages of ripeness, and multiple viewing angles. "
            "Test with both close-up and distant views of cherry clusters."
        )
job_id = submit_job(API_KEY, s3_uri, target_classes, context)
```

#### API Parameters Explained

- **s3_model_package_uri**: The S3 URI returned from the presigned URL response, which identifies your uploaded model package
- **target_classes**: Array of class names that your model should be tested on. Must be a subset of the names in the CLASS_MAPPING dictionary from your inference.py
- **context_description**: Natural language description of the contexts where you want your model evaluated (see below)

### Creating Effective Context Descriptions

When submitting an evaluation job, the `context_description` parameter is crucial as it determines what scenarios your model will be tested in. Here are guidelines for creating effective context descriptions:

#### Best Practices

- **Be specific and detailed** about the environments where your model will be deployed
- Include information about **lighting conditions** (bright sunlight, dim indoor, night, etc.)
- Specify **background variations** that might be present (foliage, urban settings, etc.)
- Mention **object variations** (size, color, orientation, partial occlusion)
- Describe **camera angles** and distances that are relevant
- Include **weather conditions** if applicable (rain, fog, snow)

#### Examples

**Basic description:**
```
"Test my cherry detection model."
```

**Improved description:**
```
"Evaluate cherry detection in outdoor orchards with varying lighting conditions including direct sunlight, overcast, and dawn/dusk. Include scenarios with cherries partially hidden by leaves, different stages of ripeness, and multiple viewing angles. Test with both close-up and distant views of cherry clusters."
```

The more specific your description, the more targeted and valuable your evaluation results will be.

### Step 5: Check Job Status and Get Results

Monitor the job status and retrieve results when complete:

> ⏱️ **Processing Time Note**: APEX creates custom test images and thoroughly evaluates your model against them in various contexts, which can take time (typically 20-60 minutes depending on model complexity). Rather than using the polling approach shown below, you may want to implement your own logic to check status at longer intervals or build a notification system.

```python
import time

def check_job_status(api_key, job_id):
    headers = {"x-api-key": api_key}
    while True:
        response = requests.get(
            "https://api.yrikka.com/v1/job-status", 
            headers=headers, 
            params={"thread_id": job_id}
        )
        
        response_data = response.json()
        status = response_data.get("status")
        message = response_data.get("message")
                
        if status == "SUCCESS":
            print("Evaluation completed!")
            return response_data.get("results") # Return evaluation metrics
        elif status in ["FAIL", "ERROR"]:
            print(f"Job failed: {message}")
            return None
            
        print("Still processing, checking again in 5 minutes...")
        time.sleep(300)

# Usage
results = check_job_status(API_KEY, job_id)
if results:
    # Print the complete results object - structure may change over time
    print("Results:", results)
```

## Example Response

When using the `/job-status` endpoint with a completed job, you'll receive a response like:

```json
{
  "status": "SUCCESS",
  "message": "Evaluation completed successfully.",
  "results": {
    "mAP": 0.87,
    "precision": 0.92,
    "recall": 0.83,
    "f1_score": 0.89,
    "TODO":0.0,
    "total_test_images": 250,
    "evaluation_timestamp": "2023-06-15T14:30:45Z"
  }
}
```

## Troubleshooting

Common issues and solutions:

- **Missing functions**: Ensure inference.py implements all required functions
- **Class mapping**: Ensure your CLASS_MAPPING includes all target_classes
- **Size limits**: Keep your model package under 4GB (packages larger than 4GB will be rejected)

## Need Help?

- API Documentation: [APEX API Documentation](https://docs.yrikka.com/apex)
- Contact Support: support@yrikka.com
