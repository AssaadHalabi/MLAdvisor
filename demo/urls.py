from .views import image_demo, text_demo
from django.urls import path
from home.dash_apps.finished_apps import textdemo, imagedemo, tabulardemo

urlpatterns = [
    path('image_demo/', image_demo, name='image_demo'),
    path('text_demo/', text_demo, name='text_demo'),
]