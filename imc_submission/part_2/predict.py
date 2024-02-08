"""
    Script: predict.py
    Author: Frederick Atisai.
    Desc: Use a trained deep neural network to predict the class for an image along with associated probabilities.
    
    Args:
        - path/to/image: the full path to input image.
        - model: saved keras model.
    
    Options:
        - top_k: return top K most likely classes along with associated probabilities.
        - category_names: Path to a JSON file mapping labels to category names.
"""

# Import TensorFlow.
import tensorflow as tf
import tensorflow_hub as hub

import argparse # Argparse to parse cmd-line args, options and sub-commands.
import numpy as np

from PIL import Image # Module for loading images from files.

# Import utility functions
from utils import process_image, get_label_map

# Ignore some warnings that are not relevant.
import warnings
warnings.filterwarnings('ignore')

# Create argument parser object.
parser = argparse.ArgumentParser(description = "Predict class of image along with probability of belonging to class.")

# Add positional arguments.
parser.add_argument("image_path", help = "path/to/image")
parser.add_argument("model", help = "saved keras model")

# Add optional arguments.
parser.add_argument("--top_k", type = int, help = "return top K most likely classes with associated probabilities")
parser.add_argument("--category_names", help = "path to JSON file mapping labels to category names")

# Parse arguments.
args = parser.parse_args()

# Load the model.
model = tf.keras.models.load_model(args.model, custom_objects = {'KerasLayer':hub.KerasLayer}, compile = False)

# Load the image and convert to NumPy array
image = Image.open(args.image_path)
image = np.asarray(image)
    
# Process the image and add an extra dimension to match models' expected input.
processed_image = process_image(image)
processed_image_expanded = np.expand_dims(processed_image, axis = 0)
    
# Make predictions.
ps = model.predict(processed_image_expanded)
ps = ps.squeeze() # Remove extra dim from NumPy ndarray returned from model.predict.

# Variables.
class_names = None
result = None

# Get label map if option included.
if args.category_names:
    class_names = get_label_map(args.category_names)

# Get top K most likely classes along with associated probabilities if option included.
if args.top_k:
    # Get classes (indices) of top K predictions.
    top_k_classes = np.argpartition(ps, args.top_k * (-1))[args.top_k  * (-1):]
    
    # Get probabilities corresponding to indices.
    top_k_probs = ps[top_k_classes]
    
    # Convert integer classes to category names if class_names available.
    if (class_names is not None): top_k_classes = [class_names[str(i+1)] for i in top_k_classes]
    
    # Format result for printing.
    result = "".join("Probabilities: {}\nClasses: {}".format(top_k_probs, top_k_classes))

# args.top_k is None
else:
    # Get most likely label and probability
    label = np.argmax(ps)
    prob = ps[label]
    
    # Convert integer class to category name if class_names available.
    if (class_names is not None): label = class_names[str(label + 1)]
    
    # Format result for printing
    result = "".join("Probability: {}\nClass: {}".format(prob, label))
    
# Print result
print(result)