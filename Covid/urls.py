from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views

urlpatterns = [
    path('',views.home,name='homepage'),
    path('predict/',views.predict,name='predict'),
    path('author/',views.author,name='author'),
]

