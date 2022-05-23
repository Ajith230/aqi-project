"""Demo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from smalldemo import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.view),
    path('new',views.indx),
    path('register',views.regster),
    path('homepage',views.homepage),
    path('logout',views.user_logout),
    path('reg',views.register2),
    path('data1',views.data1),
    path('data2',views.data2),
   # path('data1/value',views.value),
    path('value',views.value),
    path('range',views.range),
]

urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
