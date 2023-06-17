from django.urls import re_path, include, path
from rest_framework.routers import DefaultRouter
from . import views

from apps.endpoints.views import EndpointViewSet
from apps.endpoints.views import MLAlgorithmViewSet
from apps.endpoints.views import MLAlgorithmStatusViewSet
from apps.endpoints.views import MLRequestViewSet
from apps.endpoints.views import PredictView

from django.conf import settings
from django.conf.urls.static import static

router = DefaultRouter(trailing_slash=False)
router.register(r'endpoints', EndpointViewSet, basename="endpoints")
router.register(r'mlalgorithms', MLAlgorithmViewSet, basename="mlalgorithms")
router.register(r'mlalgorithmstatuses', MLAlgorithmStatusViewSet, basename="mlalgorithmstatuses")
router.register(r'mlrequests', MLRequestViewSet, basename="mlrequests")

urlpatterns = [
    re_path(r'^api/v1/', include(router.urls)),
    path('food/', views.listFoods, name="foods"),
    path('upload/', views.upload, name="upload"),
    path('', views.home, name="home"),
    path('upload/result/', views.Prediction, name="result"),
    re_path(
        r"^api/v1/(?P<endpoint_name>.+)/predict$", PredictView.as_view(), name="predict"
    )
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
