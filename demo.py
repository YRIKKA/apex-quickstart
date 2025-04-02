"""
This script demonstrates how to package an object detection model directory, upload it to Yrikka, and submit context-based evaluation job.

Directory Structure:
yolo_v9_tiny/
    ├── model.pt          # Model weights
    ├── inference.py      # Inference script
    └── manifest.json     # Metadata and configuration specifiing the names of the inference script and model weights files

Overall Flow:
1. Create a tarball from the model directory.
2. Request a presigned URL from the Yrikka API to upload the tarball.
3. Upload the tarball using the presigned URL.
4. Submit a job to the Yrikka API with the uploaded model package.
5. Periodically check the status of the submitted job until it completes.
6. Print the results of the job once it is finished.

Note: 
- Ensure that the YRIKKA_API_KEY environment variable is set with your API key before running the script.
- Ensure that your model directory contains the required files.
- Ensure that your model_package tarball is less than 4 GB.
"""

import os
import tarfile
import time
import requests

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("YRIKKA_API_KEY")

# Configuration
my_model_dir = "examples/yolo_v9_tiny" # Directory containing the model files (model.pt, inference.py, manifest.json)
output_filename = "model_package.tar.gz" # Name of the temporary tarball to be created

# URLs for Yrikka API endpoints
presigned_url = "https://api.yrikka.com/v1/presigned"
submit_job_url = "https://api.yrikka.com/v1/submit-job"
job_status_url = "https://api.yrikka.com/v1/job-status"

# Headers for API requests
headers = {
    "x-api-key": api_key,
    "Content-Type": "application/json"
}

# Function to create a tarball from the model directory
def create_tarball(directory, output_file):
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(directory, arcname=os.path.basename(directory)) # Assuming the directory includes all required files
    print(f"Created tarball '{output_file}' from directory '{directory}'")

# Function to upload the tarball to the presigned URL
def upload_tarball(upload_url, file_path):
    with open(file_path, "rb") as f:
        response = requests.put(upload_url, data=f)
    response.raise_for_status()
    print("Pre-signed upload succeeded!")

# Function to request a presigned URL for uploading the tarball
def get_presigned_url():
    response = requests.get(presigned_url, headers=headers)
    response.raise_for_status()
    return response.json()["upload_url"], response.json()["s3_uri"]

# Function to submit a job to the Yrikka API
def submit_job(model_package_uri):
    data = {
        "s3_model_package_uri": model_package_uri,
        "target_classes": ["snowboard", "skateboard", "frisbee"], # Classes to be tested (from the CLASS_NAMES in your inference script)
        "context_description": (
            "Test the Yolo v9 Tiny model at skate parks, snow resorts, and public parks during different times of day and weather conditions. "
            "Generate test scenarios with people actively using the equipment as well as scenes where multiple objects are scattered "
            "around at various distances from the camera."
        )
    }
    response = requests.post(submit_job_url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["job_id"]

# Function to periodically check the status of the submitted job
def check_job_status(job_id):
    while True:
        response = requests.get(job_status_url, headers=headers, params={"job_id": job_id})
        response.raise_for_status()
        
        response_data = response.json()
        status = response_data.get("status")
        message = response_data.get("message")
        
        print(f"Response: {response_data}")
        if status == "SUCCESS":
            print("Evaluation completed successfully!")
            return response_data.get("results") # Return the full results object
        elif status in ["FAIL", "ERROR"]:
            print(f"Job failed with message: {message}")
            return None
        
        print("Still processing. Checking again in 5 minutes...")
        time.sleep(300)


if __name__ == "__main__":
    # Create tarball from model directory
    create_tarball(my_model_dir, output_filename)  

    # Get presigned URL for upload
    upload_url, model_package_uri = get_presigned_url()  

    print(f"Upload URL: {upload_url}")
    print(f"Model package URI: {model_package_uri}")

    # Upload the tarball
    upload_tarball(upload_url, output_filename)  

    # Submit the job to Yrikka API
    job_id = submit_job(model_package_uri)  
    print(f"Submitted new job with ID: {job_id}")
    
    # Check the job status
    results = check_job_status(job_id)  
    
    if results:
        print("\n=== EVALUATION RESULTS ===")
        print(results)
    else:
        print("No results available.")
