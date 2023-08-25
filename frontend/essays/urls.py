from django.urls import path

from .views import *
from . import views

urlpatterns = [
    path("", EssayList.as_view(), name="list"),
    path("<pk>/", EssayDetail.as_view(), name="detail"),
    path("create", EssayCreate.as_view(), name="create"),
    path("theme/create", ThemeCreate.as_view(), name="create_theme"),
    path("<pk>/correct", EssayCorrect.as_view(), name="correct_essay"),

    path("themes", ThemeList.as_view(), name="theme_list"),
    path("themes/<pk>/", ThemeDetail.as_view(), name="theme_detail"),
    path('',Home.as_view(), name="home" ),

]