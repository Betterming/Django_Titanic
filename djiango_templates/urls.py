"""djiango_templates URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from app01 import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path(r'', views.home_ordinary),
    path(r'register/', views.register),
    path(r'login/', views.login),
    path(r'home_ordinary/', views.home_ordinary),
    path(r'home_coder/', views.home_coder),
    path(r'home_manager/', views.home_manager),
    path(r'home_manager_add/', views.home_manager_add),
    path(r'home_manager_find/', views.home_manager_find),
    path(r'home_manager_file/', views.home_manager_file),
    path(r'home_manager_model/', views.home_manager_model),
    path(r'delete/', views.delete),
    path(r'model1/', views.model1),
    path(r'model2/', views.model2),
    path(r'model3/', views.model3),
    path(r'model4/', views.model4),
    path(r'model5/', views.model5),
    path(r'model6/', views.model6),
    path(r'ajax_recv/', views.Ajax_recv),
    path(r'grid_search/', views.grid_search),
    path(r'model2/test/', views.test),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



