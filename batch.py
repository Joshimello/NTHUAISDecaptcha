from time import time
import tensorflow as tf
import urllib.request
import numpy as np
import requests
import cv2
from bs4 import BeautifulSoup
import os

model = tf.lite.Interpreter(model_path='model.tflite', num_threads=4)

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
  for img in images:
    img = pad(img) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.allocate_tensors()
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])
    answer = np.argmax(prediction[0])
    answers.append(answer)
  
  answer = ''.join(map(str, answers))
  return answer

def get_one():
  url = 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/'
  response = requests.get(url)
  soup = BeautifulSoup(response.content, "html.parser")
  img_tag = soup.find_all("img", src=lambda x: x and "auth_img.php?pwdstr=" in x)
  if img_tag:
    img_url = img_tag[0]['src']
    answer = uwu(url + img_url)
    img_response = requests.get(url + img_url)
    with open(f'images/{answer}.png', 'wb') as f:
      f.write(img_response.content)
      
def main():
  if not os.path.exists('images'):
    os.makedirs('images')
  for i in range(100):
    get_one()
    print(f'Got {i + 1} images')
    
main()