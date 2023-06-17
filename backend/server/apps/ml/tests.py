from django.test import TestCase
import numpy as np
import tensorflow as tf
from apps.ml.ingredients_classifier.conv_net import ConvolutionalClassifier
from tensorflow.keras import utils
import os
import inspect
from apps.ml.registry import MLRegistry


class MLTests(TestCase):
    def test_conv_net(self):
        input_data = tf.keras.utils.load_img(os.path.abspath('apps/ml/images/pineapple.jpg'),
                                             target_size=(180, 180),
                                             keep_aspect_ratio=True)

        input_data = utils.img_to_array(input_data)
        input_data = np.array([input_data])

        convnet = ConvolutionalClassifier()
        response = convnet.compute_prediction(input_data)
        print(response)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('Pineapple', response['label'])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "ingredients_classifier"
        algorithm_object = ConvolutionalClassifier()
        algorithm_name = "convolutional classifier"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Carina"
        algorithm_description = "Convolutional Classifier with simple pre- and post-processing"
        algorithm_code = inspect.getsource(ConvolutionalClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)