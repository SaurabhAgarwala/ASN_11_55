from django.conf.urls import url
from . import views

app_name = 'proctor'

urlpatterns = [
   url(r'^violation/$', views.violation, name="violation"),
]