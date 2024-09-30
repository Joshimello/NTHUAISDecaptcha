import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import urllib.request
import numpy as np
import requests
import cv2
import threading
import re

counter = 0
counter_lock = threading.Lock()

def load(url):
  response = urllib.request.urlopen(url)
  array = np.array(bytearray(response.read()), dtype=np.uint8)
  img = cv2.imdecode(array, -1)
  return img

def pad(image, size = (30, 30)):
  h, w = image.shape[:2]
  dh, dw = size[0] - h, size[1] - w
  top, left = dh // 2, dw // 2
  bottom, right = dh - top, dw - left
  padding = ((top, bottom), (left, right)) + ((0, 0),) * (image.ndim - 2)
  return np.pad(image, padding, 'constant')

def process(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
  contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
  subimages = []
  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    subimage = image[y:y+h, x:x+w]
    subimages.append(subimage)
  return subimages

def uwu(url):
  image = load(url)
  images = process(image)

  answers = []
  for i, img in enumerate(images):
    img = pad(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    model = tf.lite.Interpreter(model_path='../models/cnn_seg/model.tflite', num_threads=4)
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.allocate_tensors()
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])
    answer = np.argmax(prediction[0])
    answers.append(answer)
  
  answer = ''.join(map(str, answers))
  return answer, image

def get_url():
  return 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/JH/mod/auth_img/auth_img.php?ACIXSTORE=aohmo9gaf8m9fmieb9rqp2ghd1'

def get_one():
  global counter
  answer = ''
  image = None
  url = get_url()
  while (len(answer) != 3):
    answer, image = uwu(url)
  with open(f'images/{answer}.png', 'wb') as f:
    f.write(cv2.imencode('.png', image)[1])
    with counter_lock:
      counter += 1
      print(f'#{counter} [{answer}]')
      
def main():
  if not os.path.exists('images'):
    os.makedirs('images')
    
  num_images = 200
  
  threads = []
  for i in range(num_images):
    thread = threading.Thread(target=get_one)
    thread.start()
    threads.append(thread)
    
  for thread in threads:
    thread.join()
  
main()