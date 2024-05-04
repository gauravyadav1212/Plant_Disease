from fastapi import FastAPI, File, UploadFile
import unicorn 
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

app = FastAPI()


MODEL_PATH = r"C:\Users\Gaurav\OneDrive\Desktop\Potato_disease\saved_models\1"

# Load the TensorFlow SavedModel as an inference-only layer
MODEL = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    #print(image)
    img_batch = np.expand_dims(image, 0)  # To add one more dimension to the one dimension array

    predictions = MODEL(img_batch)  
    print(predictions)
    predictions_tensor = predictions['dense_1']  # Extract the tensor from the dictionary
    predictions_array = predictions_tensor.numpy()  # Convert the tensor to a NumPy array
    print(predictions_array)
    max_index = np.argmax(predictions_array)  # Get the index of the maximum value
    predicted_class = CLASS_NAMES[max_index]  # Get the corresponding class name
    confidence = np.max(predictions_array) * 100 # Get the confidence
    print("Predicted class: " + predicted_class)
    print("Confidence: "+ str(confidence) + " %")

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)