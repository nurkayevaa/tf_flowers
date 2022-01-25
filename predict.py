# Import TensorFlow 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# Ignore some warnings that are not relevant (you can remove this if you prefer)
import warnings
warnings.filterwarnings('ignore')
# TODO: Make all other necessary imports.
import numpy as np

tfds.disable_progress_bar()
import logging
import json
import matplotlib.pyplot as plt
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
from PIL import Image


def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image

def predict(image_path, model, top_k=5):
    image_path = image_path
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    ps = model( np.expand_dims(processed_test_image, axis=0), training=False)
    return np.array(tf.gather(ps[0],indices=np.argpartition(ps[0], -5)[-5:])), list(np.argpartition(ps[0], -5)[-5:])

# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an arguments
parser.add_argument('path', type=str, )
parser.add_argument('model', type=str, )
parser.add_argument('--top_k', type=int, required=False)
parser.add_argument('--category_names', type=str, required=False)
# Parse the argument
args = parser.parse_args()


# reloaded_SavedModel = tf.saved_model.load(args.model)

image_size=224
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size,3))
feature_extractor.trainable = False
model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(640, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(320, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(160, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(102, activation = 'softmax')
])
model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.load_weights(args.model)


def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image

def predict(image_path, model, top_k):
    image_path = image_path
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    ps = model( np.expand_dims(processed_test_image, axis=0), training=False)
    return np.array(tf.gather(ps[0],indices=np.argpartition(ps[0], -top_k)[-top_k:])), list(np.argpartition(ps[0], -top_k)[-top_k:])

print(args)
# importing the module
import json
 
# Opening JSON file
def read_json():
    path = args.category_names if args.category_names is not None else 'label_map.json'
    with open(path) as json_file:
        class_names = json.load(json_file)
    return class_names
    

class_names = read_json()
top_k = args.top_k if args.top_k is not None else top_k
 
probs, classes = predict(args.path, model, top_k)
print(classes)
print( [class_names[str(k+1)] for k in classes])