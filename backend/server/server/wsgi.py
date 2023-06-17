"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

import inspect
from apps.ml.registry import MLRegistry
from apps.ml.ingredients_classifier.conv_net import ConvolutionalClassifier

try:
    registry = MLRegistry()

    rf = ConvolutionalClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="ingredients_classifier",
                            algorithm_object=rf,
                            algorithm_name="convolutional classifier",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Carina",
                            algorithm_description="Convolutional Classifier with preprocessing",
                            algorithm_code=inspect.getsource(ConvolutionalClassifier))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
