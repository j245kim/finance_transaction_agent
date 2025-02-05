from django.urls import path, include
from .views import authView, home, manual, introduction, logout_view
from . import views

urlpatterns = [
 path("", home, name="home"),
 path("signup/", authView, name="authView"),
 path("accounts/", include("django.contrib.auth.urls")),
 path('profile/edit/', views.profile_edit, name='profile_edit'), 
 path ("manual/", views.manual, name="manual"),
 path ("introduction/", views.introduction, name="introduction"),
 path ("logout/", views.logout_view, name="logout"),
]