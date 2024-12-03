import os
from keras_cnn import create_dataset
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from tensorflow.keras.preprocessing import image

sign_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
}

img_size = (56, 56)

last_conv_layer_name = "conv2d_1"
last_model_layer = "dense_1"

def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    _ = model(img_array)
    grad_model = keras.models.Model(
        model.inputs, model.get_layer(last_conv_layer_name).output
    )
    class_input = tf.keras.Input(shape=model.get_layer(last_conv_layer_name).output.shape[1:])
    x = class_input
    for layer in ['max_pooling2d', 'flatten', 'dense', 'dense_1']:
        x = model.get_layer(layer)(x)
    class_model = keras.models.Model(class_input, x)

    img_array = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        last_conv_layer_output = grad_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = class_model(last_conv_layer_output)

        if pred_index is None:
            pred_index = np.argmax(preds)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        print("Gradientes nulos")
        return
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

dataset = pd.read_csv("images/dataset/sign_mnist_big.csv")

for label in sign_dict.keys():
    if (label != 9) and (label != 25):
        sign = sign_dict[label]
        # shuffle
        p = np.random.permutation(len(dataset))
        dataset = dataset.iloc[p]
        img = dataset[dataset.iloc[:, 0] == label].iloc[0, 1:]
        img = img.to_numpy().reshape(56, 56, 1)
        img_array = np.expand_dims(img / 255.0, axis=0)

        model = tf.keras.models.load_model("models/final_model.keras", compile=False)
        pred = model.predict(img_array)
        pred_index = np.argmax(pred)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        heatmap = np.uint8(255 * heatmap)

        plt.imshow(heatmap)
        plt.title("Predicción: {}".format(sign_dict[pred_index]) + " Real: {}".format(label))
        plt.show()

        jet = plt.get_cmap('jet')
        jet_colors = jet(np.arange(256))[:, :3]

        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((56, 56))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * 0.5 + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)
        plt.imshow(superimposed_img)
        plt.title("Predicción: {}".format(sign_dict[pred_index]) + " Real: {}".format(sign_dict[label]))
        plt.axis('off')
        plt.close()

