from django.conf.urls import url
from . import views

app_name = 'proctor'

urlpatterns = [
   url(r'^violation/$', views.get_violation, name="get_violation"),
   url(r'^test/$', views.test_view, name="test"),
   # url(r'^student-exam/$', views.student_exam, name="student-exam"),
]