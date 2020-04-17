from . import views
from django.conf.urls import url

urlpatterns = [
    url(r'^home/?', views.Home.as_view(), name="home"),
    url(r'^upload/?', views.upload, name="upload"),
    url(r'^db_view/?', views.db_view, name="db_view"),
    url(r'^visualization/?', views.visualize, name="visualization"),
]
