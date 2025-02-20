from django.urls import path, include
from .views import authView, home, manual, aboutus, logout_view, copyright, trading
from . import views

urlpatterns = [
 path("", home, name="home"),
 path("signup/", authView, name="authView"),
 path("accounts/", include("django.contrib.auth.urls")),
 path('profile/edit/', views.profile_edit, name='profile_edit'), 
 path ("manual/", views.manual, name="manual"),
 path ("aboutus/", views.aboutus, name="aboutus"),
 path ("logout/", views.logout_view, name="logout"),
 path ("terms/", views.terms, name="terms"),
 path ("copyright/", views.copyright, name="copyright"),
 path ("survey/", views.survey, name="survey"),
 path ("predict/", views.predict, name="predict"),
 path ("trading/", views.trading, name="trading"),
]