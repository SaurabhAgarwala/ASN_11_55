from django.conf.urls import url
from . import views

app_name = 'proctor'

urlpatterns = [
   url(r'^violation/$', views.violation, name="violation"),
   url(r'^test/$', views.test_view, name="test"),
   # url(r'^student-exam/$', views.student_exam, name="student-exam"),
]