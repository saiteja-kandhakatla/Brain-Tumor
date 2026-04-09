from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from predict import CLASS_NAMES, IMG_SIZE, load_model

TEST_DIR = "Brain-Tumor/Testing"
BATCH_SIZE = 32


@lru_cache(maxsize=1)
def evaluate_model():
    test_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_data = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    model = load_model()
    predictions = model.predict(test_data, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    confusion = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    confusion_df = pd.DataFrame(confusion, index=CLASS_NAMES, columns=CLASS_NAMES)

    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        zero_division=0,
    )

    class_report_df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Precision": [round(value * 100, 2) for value in per_class_precision],
            "Recall": [round(value * 100, 2) for value in per_class_recall],
            "F1 Score": [round(value * 100, 2) for value in per_class_f1],
            "Support": support,
        }
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_df,
        "class_report": class_report_df,
    }
