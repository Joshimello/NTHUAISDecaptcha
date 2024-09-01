# NTHU AIS Decaptcha

Solving NTHU AIS captchas with deep learning models, implemented as APIs.

> [!TIP]
> This project was made for <a href="https://nthumods.com" target="_blank">NTHUMods</a>!  
> Go check it out! It's <a href="https://github.com/nthumodifications/courseweb" target="_blank">open source</a> too!

## Methods

1. [cnn_seg]: Segmentation + CNN  

    This method involves segmenting the CAPTCHA image into individual characters and then using a Convolutional Neural Network (CNN) to recognize each character.

2. [cnn_ctc]: CNN + CTC Layer

    This method involves using a CNN trained with a Connectionist Temporal Classification (CTC) loss function to recognize the entire CAPTCHA image at once.

Depending on the use case, different methods may be more suitable.

## Getting Started

To get started with running the model API, you can use the provided run.sh script, which is set up to run the cnn_ctc.py model by default.

### Prerequisites

Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

Or if you are on Windows:

```bash
pip install -r requirements-windows.txt
```

### Running the model

To run the cnn_ctc model as a fetchable API:

```bash
./run.sh
```

Or if you want to run the model manually:

```bash
sudo gunicorn cnn_ctc:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000
```

Or if you are on Windows:

```bash
fastapi dev cnn_ctc.py
```

Or for Windows production:

```bash
fastapi run cnn_ctc.py
```

You may also run the cnn_seg model by changing the script to cnn_seg.py.

## API Usage

Make a GET request to the api, passing in the url of the image:

```
https://example.api:5000?url=IMAGEURL
```

Or visit the fastAPI docs at:

```
https://example.api:5000/docs
```

The API will return the predicted text of the image in plain text format.

## Pre-trained Model Usage (cnn_seg) 

Two formats of the image recognition model is available in the /models folder  
These models are for single digit recognition, separation is still required

- Tensorflow [.h5] model format
- TFLite [.tflite] format

Example usage for the TFLite model:

```py
import tflite_runtime.interpreter as tflite
model = tflite.Interpreter(model_path='models/cnn_seg/model.tflite')
# Or
import tensorflow as tf
model = tf.lite.Interpreter(model_path='models/cnn_seg/model.tflite')
```

```py
input_details = model.get_input_details()
output_details = model.get_output_details()
model.allocate_tensors()
model.set_tensor(input_details[0]['index'], img)
model.invoke()
prediction = model.get_tensor(output_details[0]['index'])
answer = np.argmax(prediction[0])
print(''.join(map(str, answers)))
```

You may view usage examples in the notebooks, scripts, or cnn_seg.py file

## CNN_CTC Usage (Pre-trained Model)

There are a few variations of the CTC model available in the /models folder  
Some options containing the CTCLoss layer (if you want to train further)

- Tensorflow [.keras] model format (with CTC variation)
- Tensorflow [.h5] model format (with CTC variation)
- Tensorflow [saved_model] format
- TFLite [.tflite] format (quantized model available)

Usage same as above, but with the respective model format  
Note: The output of this model has been labeled and requires decoding. Implementations can be found in the notebooks, scripts, or cnn_ctc.py file

## Usage (Training/Notebook)

To train your own models or experiment with different architectures:

1. Navigate to the notebooks folder.
2. Open the respective ipynb file in Google Colab.
3. Load the data from the training folder if necessary.
4. Have fun experimenting!  

Note: The full experimental process and training was done on Google Colab, I am unabled to guarantee the same results on local machines.  

Full training processes can be found in the notebooks folder.  
All data used for training can be found in the training folder.  
Data generated (300.zip, 1000.zip) for cnn_ctc training used was generated by the cnn_seg model, then handpicked for correctness.

## Benchmarking

You can benchmark the performance of the models using the scripts in the benchmarks folder. The benchmarks will help you evaluate the speed and accuracy of each model when deployed as an API.

Results will be published soon.

## Contributing

Contributions are always welcomed! If you have any ideas or improvements, feel free to fork the repository, make your changes, and submit a pull request. Also, feel free to open an issue if you have any suggestions or feedback.

Note: I am not a professional machine learning engineer, so any help or advice is greatly appreciated.

## References

- [Keras - Automatic Speech Recognition using CTC](https://keras.io/examples/audio/ctc_asr/)
- [Hugging Face - keras-io/ocr-for-captcha](https://huggingface.co/keras-io/ocr-for-captcha)
- [Not a reference, but I wrote this for fun for a image processing class while working on cnn_seg](https://github.com/Joshimello/NTHUAISDecaptcha/blob/main/extras/cnn_seg.pdf)

## Special Thanks

[Audrey](https://github.com/audreych23) for helping me brain everything  
[Chew](https://github.com/ImJustChew) for helping me debug everything  

## License

MIT
