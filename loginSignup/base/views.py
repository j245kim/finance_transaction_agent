from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .forms import EncryptedUserCreationForm  

from pymongo import MongoClient
from django.contrib.auth.hashers import make_password 

from dotenv import load_dotenv
import os 

from .forms import ProfileForm 

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

@login_required
def profile_edit(request):
    user = request.user  # 현재 로그인한 사용자
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=user)  # 기존 정보를 채운 폼
        if form.is_valid():
            form.save()  # 유효한 폼은 저장

            # mongodb 업데이트트
            user_info = {
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'password': user.password,  
            }

            # 옛날 파일 없에기
            collection.delete_one({'username': user.username})  # 삭제
            collection.insert_one(user_info)  # 신규 정보 삽입

            return redirect('/')  # 홈으로 다시
    else:
        form = ProfileForm(instance=user)  # GET 요청 시, 기존 사용자 정보로 폼을 채움

    return render(request, 'registration/profile_edit.html', {'form': form})

def manual(request):
    return render(request, "registration/function.html", {})

def introduction (request):
    return render(request, "registration/introduction.html", {})