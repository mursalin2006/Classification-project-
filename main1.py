from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

load_model = tf.saved_model.load("../models/1")
MODEL = load_model.signatures["serving_default"]

CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((256, 256))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    
    predictions = MODEL(tf.constant(image_array, dtype=tf.float32))
    output_tensor = list(predictions.values())[0]
    probabilities = output_tensor.numpy()[0]
    
    predicted_index = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    return {
        "class": CLASS_NAMES[predicted_index],
        "confidence": confidence
    }