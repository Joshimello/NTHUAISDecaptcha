from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import tensorflow as tf
import numpy as np
import requests

app = FastAPI()
model = tf.lite.Interpreter(model_path='models/cnn_ctc/model_quantized.tflite', num_threads=4)

characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, mask_token=None)
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def preprocess_image(img_content, img_height, img_width):
  img = tf.io.decode_png(img_content, channels=1) 
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [img_height, img_width])
  img = tf.transpose(img, perm=[1, 0, 2])
  img = tf.expand_dims(img, axis=0)
  return img

def decode_batch_predictions(pred):
  input_len = np.ones(pred.shape[0]) * pred.shape[1]
  results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :6]
  output_text = []
  for res in results:
    res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
    output_text.append(res)
  return output_text

@app.get('/', response_class=PlainTextResponse)
async def uwu(url):
  response = requests.get(url)
  imgs = preprocess_image(response.content, 32, 104)
  
  input_details = model.get_input_details()
  output_details = model.get_output_details()
  model.allocate_tensors()
  model.set_tensor(input_details[0]['index'], imgs)
  model.invoke()
  preds = model.get_tensor(output_details[0]['index'])
    
  preds_texts = decode_batch_predictions(preds)
  return preds_texts[0]