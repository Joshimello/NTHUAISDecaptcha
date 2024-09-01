from tensorflow import keras
import tensorflow as tf
import numpy as np
import requests 
import matplotlib.pyplot as plt
import threading
import re
import os

counter = 0
counter_lock = threading.Lock()

model = keras.models.load_model('../models/cnn_ctc/model.h5', compile=False)

def preprocess_image(img_content, img_height, img_width):
    img = tf.io.decode_png(img_content, channels=1) 
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)
    return img

characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :6
    ]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

char_to_num = keras.layers.StringLookup(
    vocabulary=characters, mask_token=None
)
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def predict_with_model(imgs):
    return model.predict(imgs)

def uwu(url):
    response = requests.get(url)
    imgs = preprocess_image(response.content, 32, 104)
    # preds = model.predict(imgs)
    preds = predict_with_model(imgs)
    preds_texts = decode_batch_predictions(preds)
    return preds_texts[0]

def get_url():
  url = 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/'
  response = requests.get(url)
  html_text = response.text
  match = re.search(r'<img\s+src=auth_img\.php\?pwdstr=([0-9\-]+)', html_text)
  return url + 'auth_img.php?pwdstr=' + match.group(1)

def get_one():
  global counter
  url = get_url()
  answer = uwu(url)
  img_response = requests.get(url)
  with open(f'images/{answer}.png', 'wb') as f:
    f.write(img_response.content)
    with counter_lock:
      counter += 1
            
def main():
  if not os.path.exists('images'):
    os.makedirs('images')
    
  num_images = 100
  
  threads = []
  for i in range(num_images):
    thread = threading.Thread(target=get_one)
    thread.start()
    threads.append(thread)
    
  for thread in threads:
    thread.join()
  
main()