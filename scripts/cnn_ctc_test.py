from tensorflow import keras
import tensorflow as tf
import numpy as np
import requests 
import matplotlib.pyplot as plt

url = 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/JH/mod/auth_img/auth_img.php?ACIXSTORE=fsi5992afd4s8abo2fk029as60'
model = keras.models.load_model('../models/cnn_ctc/model.h5', compile=False)

characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

char_to_num = keras.layers.StringLookup(
    vocabulary=characters, mask_token=None
)
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def preprocess_image(img_content, img_height, img_width):
    img = tf.io.decode_png(img_content, channels=1) 
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)
    return img

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :3
    ]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

response = requests.get(url)
imgs = preprocess_image(response.content, 32, 104)
preds = model.predict(imgs)
preds_texts = decode_batch_predictions(preds)
print(preds_texts[0])