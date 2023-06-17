import os

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils


class ConvolutionalClassifier:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.model = joblib.load(path_to_artifacts + "custom-resnet.joblib")
        self.classes = joblib.load(path_to_artifacts + "class_names.joblib")
        self.NUM_CLASSES = 28

    def preprocessing(self, input_data):
        return input_data

    def postprocessing(self, output_data):
        result = {}
        add = True
        for data in output_data:
            for label in equivalent_classes:
                for subclass in equivalent_classes.get(label):
                    if data[0] == subclass:
                        if result.get(label) is not None:
                            result[label] = result.get(label) + data[1]
                            add = False
                        else:
                            result[label] = data[1]
                            add = False
            if add:
                result[data[0]] = data[1]
            add = True

        return result

    def predict(self, input_data):
        return self.model.predict(input_data)

    def compute_prediction(self, input_data):
        try:
            input_data = tf.keras.utils.load_img(os.path.abspath('media/Ingredients.jpg'),
                                                target_size=(180, 180),
                                                keep_aspect_ratio=True)

            input_data = utils.img_to_array(input_data)
            input_data = np.array([input_data])
            prediction = self.predict(input_data)

            # prediction = self.postprocessing(prediction)
            result = np.argmax(prediction[0])

            values = [(self.classes[i], prediction[0][i]) for i in range(self.NUM_CLASSES)]
            values = sorted(values, key=lambda x: x[1], reverse=True)
            print([value[0] for value in values])
            prediction = {"status": "OK", "labels": [value[0] if value[1] > 0.0005 else None for value in values]}
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
