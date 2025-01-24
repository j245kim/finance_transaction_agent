from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .forms import EncryptedUserCreationForm  

from pymongo import MongoClient
from django.contrib.auth.hashers import make_password 

from dotenv import load_dotenv
import os 

# load .env
load_dotenv()

cluster = MongoClient(os.environ.get("mongo")) # 클러스터 
db = cluster['userinfo'] # 유저 정보
collection = db["info"] # 내가 지정한 컬렉션 이름

@login_required  
def home(request):
    return render(request, "home.html", {})

def authView(request):
    if request.method == "POST":
        form = EncryptedUserCreationForm(request.POST)  
        if form.is_valid():
            user = form.save() 
            
            hashed_password = make_password(user.password)

            user_info = {
                'username': user.username,
                'email': user.email,
                'password': hashed_password, 
            }

            collection.insert_one(user_info)

            return redirect('base:login') 
    else:
        form = EncryptedUserCreationForm()
        
    return render(request, "registration/signup.html", {"form": form})