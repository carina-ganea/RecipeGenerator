import os
import json
from numpy.random import rand

from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets
from rest_framework import mixins

from apps.endpoints.models import Endpoint
from apps.endpoints.serializers import EndpointSerializer

from apps.endpoints.models import MLAlgorithm
from apps.endpoints.serializers import MLAlgorithmSerializer

from apps.endpoints.models import MLAlgorithmStatus
from apps.endpoints.serializers import MLAlgorithmStatusSerializer

from apps.endpoints.models import MLRequest
from apps.endpoints.serializers import MLRequestSerializer

from django.http import HttpResponse
from django.template import loader

from .models import Food

from django.core.files.storage import FileSystemStorage

from rest_framework import views, status
from rest_framework.response import Response

from apps.ml.registry import MLRegistry
from server.wsgi import registry

from apps.ml.ingredients_classifier.conv_net import ConvolutionalClassifier

class EndpointViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()


class MLAlgorithmViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = MLAlgorithmSerializer
    queryset = MLAlgorithm.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm=instance.parent_mlalgorithm,
                                                    created_at__lt=instance.created_at,
                                                    active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
    MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])


class MLAlgorithmStatusViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.CreateModelMixin
):
    serializer_class = MLAlgorithmStatusSerializer
    queryset = MLAlgorithmStatus.objects.all()

    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                # set active=False for other statuses
                deactivate_other_statuses(instance)



        except Exception as e:
            raise APIException(str(e))


class MLRequestViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.UpdateModelMixin
):
    serializer_class = MLRequestSerializer
    queryset = MLRequest.objects.all()


def home(request):
    template = loader.get_template("home.html")
    return HttpResponse(template.render())


def listFoods(request):
    myfoods = Food.objects.all().values()
    template = loader.get_template("all_foods.html")
    context = {
        "myfoods": myfoods,
    }
    return HttpResponse(template.render(context, request))


@csrf_exempt
def upload(request):
    template = loader.get_template("upload_image.html")
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        fss = FileSystemStorage()
        name = "Ingredients.jpg"
        if os.path.exists(os.path.join(settings.MEDIA_ROOT, name)):
            os.remove(os.path.join(settings.MEDIA_ROOT, name))

        file = fss.save(name, upload)
        file_url = fss.url(file)

        context = {
            "file_url": file_url,
        }
        return HttpResponse(template.render(context, request))
    return HttpResponse(template.render())

def Prediction(request):
    template = loader.get_template("result.html")

    from apps.endpoints.models import Ingredient, Food, Recipe
    convnet = ConvolutionalClassifier()
    response = convnet.compute_prediction(request)
    ingredients = Ingredient.objects.all()
    recipe_food_match = [0 for i in Recipe.objects.all().values()]
    recipe_food_count = [0 for i in Recipe.objects.all().values()]

    for ingr in ingredients:
        recipe_food_count[ingr.recipe.pk - 1] += 1
        if Food.objects.get(pk=ingr.food.pk).name in response["labels"]:
            recipe_food_match[ingr.recipe.pk - 1] += 1

    max_match = -1
    recipe_id = -1

    for i in range(len(recipe_food_count)):
        if max_match < recipe_food_match[i] / recipe_food_count[i]:
            max_match = recipe_food_match[i] / recipe_food_count[i]
            recipe_id = i + 1

    recipe_title = Recipe.objects.get(pk=recipe_id).title
    recipe_img = Recipe.objects.get(pk=recipe_id).image
    recipe_ingredients = []
    calories = 0

    for ingr in ingredients:
        if ingr.recipe.id == recipe_id:
            recipe_ingredients.append((Food.objects.get(pk=ingr.food.pk).name, ingr.quantity))
            calories += Food.objects.get(pk=ingr.food.pk).kcal_number * ingr.quantity / 100

    context = {
        "labels": response["labels"],
        "recipe_title": recipe_title,
        "recipe_img": recipe_img,
        "ingredients": recipe_ingredients,
        "kcal": calories
    }

    return HttpResponse(template.render(context, request))

class PredictView(views.APIView):
    def post(self, request, endpoint_name, format=None):

        algorithm_status = self.request.query_params.get("status", "production")
        algorithm_version = self.request.query_params.get("version")

        algs = MLAlgorithm.objects.filter(parent_endpoint__name=endpoint_name,
                                          status__status=algorithm_status,
                                          status__active=True)

        if algorithm_version is not None:
            algs = algs.filter(version=algorithm_version)

        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if len(algs) != 1 and algorithm_status != "ab_testing":
            return Response(
                {
                    "status": "Error",
                    "message": "ML algorithm selection is ambiguous. Please specify algorithm version."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        alg_index = 0
        if algorithm_status == "ab_testing":
            alg_index = 0 if rand() < 0.5 else 1

        algorithm_object = registry.endpoints[algs[alg_index].id]
        prediction = algorithm_object.compute_prediction(request.data)

        label = prediction["label"] if "label" in prediction else "error"
        ml_request = MLRequest(
            input_data=json.dumps(request.data),
            full_response=prediction,
            response=label,
            feedback="",
            parent_mlalgorithm=algs[alg_index],
        )
        ml_request.save()

        prediction["request_id"] = ml_request.id

        return Response(prediction)
