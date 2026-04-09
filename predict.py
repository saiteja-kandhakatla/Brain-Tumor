from functools import lru_cache

import cv2
import numpy as np
import tensorflow as tf

from gradcam import get_gradcam, overlay_heatmap

IMG_SIZE = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
LAST_CONV_LAYER = "Conv_1"
MODEL_PATH = "models/mobilenet_model.h5"


@lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def decode_image(file_bytes):
    image_array = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Unable to read the uploaded image.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(image_rgb):
    resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def predict_image(file_bytes):
    model = load_model()
    original_image = decode_image(file_bytes)
    img_array = preprocess_image(original_image)

    predictions = model.predict(img_array, verbose=0)[0]
    pred_index = int(np.argmax(predictions))
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(predictions[pred_index])

    gradcam_image = None
    if pred_class != "notumor":
        heatmap = get_gradcam(model, img_array, LAST_CONV_LAYER, class_index=pred_index)
        gradcam_image = overlay_heatmap(heatmap, original_image)

    return {
        "class_name": pred_class,
        "confidence": confidence,
        "probabilities": {
            class_name: float(score)
            for class_name, score in zip(CLASS_NAMES, predictions)
        },
        "original_image": original_image,
        "gradcam_image": gradcam_image,
    }
