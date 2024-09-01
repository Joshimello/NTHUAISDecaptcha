# NTHUAISDecaptcha

Solving NTHU AIS captcha by image recognition using convnets via tensorflow

> [!TIP]
> This project was made for <a href="https://nthumods.com" target="_blank">NTHUMods</a>!  
> Go check it out! It's <a href="https://github.com/nthumodifications/courseweb" target="_blank">open source</a> too!

## Introduction

This project is a captcha solver for the NTHU AIS captcha system.  
There is currently two different methods of recognition available.
- (cnn_seg) Recognition via segmentation and CNN
- (cnn_ctc) Recognition via CNN trained with a CTC loss function

Depending on the use case, different methods may be more suitable.

## Installation (API)

To run the recognition API

```bash
pip install -r requirements.txt
or
pip install -r requirements-windows.txt
```

Then run the script

```bash
./run.sh
```

Or run manually

```bash
sudo gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000
```

Remember to replace "main" with the name of the file containing the FastAPI app.

## Usage (API)

Make a GET request to the api, passing in the url of the image

```
https://example.api:5000?url=IMAGEURL
```

Or visit the fastAPI docs at

```
https://example.api:5000/docs
```

## CNN_SEG Usage (Pre-trained Model)

Two formats of the audio recognition model is available in the /models folder  
These models are for single digit recognition, separation is still required

- Tensorflow keras model format
- TFLite tflite format

Loading the tflite format

```py
import tflite_runtime.interpreter as tflite

model = tflite.Interpreter(model_path='model.tflite')

#or

import tensorflow as tf

model = tf.lite.Interpreter(model_path='model.tflite')


input_details = model.get_input_details()
output_details = model.get_output_details()
model.allocate_tensors()
model.set_tensor(input_details[0]['index'], x)
model.invoke()

prediction = model.get_tensor(output_details[0]['index'])
```

## CNN_CTC Usage (Pre-trained Model)

There are a few variations of the CTC model available in the /models folder  
Some options containing the CTCLoss layer (if you want to train further)



## Usage (Training/Notebook)

The full experimental process and training was done on Google Colab  
The notebook can be found in the /notebooks folder  
Data used for training can be found in the /training folder

## Contributing

Pull requests are always welcomed.

## License

MIT
