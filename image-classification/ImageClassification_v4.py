# This is an application to classify images presented to it
# It uses the 'google/vit-base-patch16-224' vision transformation model from Hugging Face
# This is a basic application to demonstrate the concept of applying a classification to an image

import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import matplotlib.pyplot as plt
import time

# Start timer
t0 = time.time()  # Add at start of process

# 0. Configuration
MODEL_ID = "google/vit-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"     # Checks if GPU is available
HIGH_DPI = 100                                              # Defines resolution of output

# 1. Define local image path (user variable)
# This path points to an image in an 'Image' folder within the same folder as the .py file
# To run locally set images in a local folder and set 'LOCAL_IMAGE_PATH' correctly
# LOCAL_IMAGE_PATH = "C:/Users/marcg/Documents/PythonProject/NLP_Transformers/Images/Test7.jpg"
LOCAL_IMAGE_PATH = "Images/Test3.jpg"

# 2. Load Model and Processor (Reusable)
try:
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID).to(DEVICE)
    print(f"Successfully loaded model: {MODEL_ID}")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()

# Define function to load a local image, runs inference, and displays the result.
def classify_local_image(image_path: str):

    # 3. Load the Image using PIL
    try:
        # Image.open() is the key function for loading local files
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {image_path}")
        print("Please check your LOCAL_IMAGE_PATH variable.")
        return
    except Exception as e:
        print(f"ERROR: Could not open/identify image file: {e}")
        return

    # 4. Process the image into a PyTorch tensor
    # The processor handles resizing (to 224x224) and normalization
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    # 5. Run Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 6. Extract Prediction
    logits = outputs.logits
    predicted_id = logits.argmax(-1).item()

    # Extract the predicted class name using the ViT model's built-in id2label
    predicted_class_name = model.config.id2label[predicted_id]

    print("\nCLASSIFICATION RESULT")
    print(f"Image Path: {image_path}")
    print(f"Predicted Class: {predicted_class_name.split(',')[0]}")  # Use first word for brevity

    # 7. Display the Image and Result
    fig = plt.figure(figsize=(6, 6), dpi=HIGH_DPI)
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(image)

    title = (
        f"ViT Classification (ImageNet):\n"
        f"Predicted: {predicted_class_name}"
    )
    ax.set_title(title, fontsize=10)
    ax.axis('off')

    # plt.tight_layout()
    plt.show()


# EXECUTE THE CLASSIFICATION
classify_local_image(LOCAL_IMAGE_PATH)

# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"Time Taken: {timetaken1:.4f} seconds")

print("Single Image Classification Complete")