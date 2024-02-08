"""
    module: utils.py
    Author: Frederick Atisai.
    
    functions:
        - process_image: Converts an resizes and image to shape (224, 224, 3).
        - get_label_map: Load mapping from label to category names.
"""

# Import TensorFlow
import tensorflow as tf

# Import JSON encoder and decoder
import json

def process_image(image):
    """
        Converts an image (in the form of a NumPy array) and
        return an image in the form of a NumPy array with shape (224, 224, 3).
    """
    image = tf.cast(image, tf.float32) # Convert to TensorFlow Tensor.
    image = tf.image.resize(image, (224, 224)) # Resize image.
    image /= 255 # Normalize pixel values.
    
    return image.numpy()

def get_label_map(label_map):
    """
        Load JSON file mapping labels to category names.
    """
    with open(label_map, 'r') as f:
        class_names = json.load(f)
    
    return class_names