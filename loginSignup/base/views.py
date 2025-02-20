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
from django.contrib.auth import logout

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
                # 'email': user.email, 그냥 이거는 나중에 넣어줍시다
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
        form = ProfileForm(request.POST)  # Form을 모델 없이 생성
        if form.is_valid():
            # 새로운 mock app key와 secret, 거래소ID 받아오기
            account_number = form.cleaned_data.get('account_number')
            mock_app_key = form.cleaned_data.get('mock_app_key')
            mock_app_secret = form.cleaned_data.get('mock_app_secret')
            거래소ID = form.cleaned_data.get('거래소ID')
            

            # mongodb 업데이트
            user_info = {
                'username': user.username,
                'password': user.password,
                '거래소ID': 거래소ID,  # 추가된 항목
                'account_number': account_number,
                'mock_app_key': mock_app_key,  # 추가된 항목
                'mock_app_secret': mock_app_secret,  # 추가된 항목
            }

            # 옛날 파일 없애기
            collection.delete_one({'username': user.username})  # 삭제
            collection.insert_one(user_info)  # 신규 정보 삽입

            return redirect('/')  # 홈으로 다시
    else:
        form = ProfileForm()  # GET 요청 시, 비어있는 폼 생성

    return render(request, 'registration/profile_edit.html', {'form': form})


def manual(request):
    return render(request, "registration/function.html", {})

def aboutus (request):
    return render(request, "registration/aboutus.html", {})

# 로그아웃웃
@login_required
def logout_view(request):
    logout(request)
    return redirect("/")

# 이용약관
def terms (request):
    return render (request, "registration/terms.html", {})

# 저작권
def copyright (request):
    return render (request, "registration/copyright.html", {})

def survey (request):
    return render (request, "registration/survey.html", {})

def trading(request):
    user = request.user 
    username = user.username
    return render(request, "registration/trading.html", {'username': username})


from io import BytesIO
import base64
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import pandas_datareader.data as reader
from datetime import date
import datetime as dt
from .forms import TickerForm
from django.shortcuts import render
from IPython.display import HTML

# 한국어
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='NanumGothic'
plt.rcParams['axes.unicode_minus'] =False

def predict(request):
    TODAY = date.today()  # 오늘 날짜를 구함
    LAST_YEAR = dt.date(TODAY.year - 1, TODAY.month, TODAY.day)

    if request.method == 'POST':
        # 사용자가 폼을 제출한 경우
        form = TickerForm(request.POST)
        if form.is_valid():
            # 폼이 유효한 경우 티커 값을 가져옴
            funds = form.cleaned_data['ticker']

            # yfinance를 사용해서 펀드 데이터를 다운로드
            try:
                data = yf.download(funds, start=LAST_YEAR, end=TODAY)
                if data.empty:
                    error_message = "해당 티커에 대한 데이터가 없습니다. 다른 티커를 입력해 주세요."
                    return render(request, 'registration/predict.html', {'form': form, 'error': error_message})
                
                # 'Adj Close'가 있으면 그걸로 계산하고, 없으면 'Close'로 계산
                if 'Adj Close' in data.columns:
                    fund_mtl = (data['Adj Close'].pct_change())  # 수정종가를 이용해 월간 수익률 계산
                elif 'Close' in data.columns:
                    fund_mtl = (data['Close'].pct_change())  # 종가를 이용해 월간 수익률 계산
                else:
                    error_message = "데이터에서 필요한 정보를 찾을 수 없습니다. 다른 티커를 입력해 주세요."
                    return render(request, 'registration/predict.html', {'form': form, 'error': error_message})

                # 월간 수익률을 계산해서 다시 월별 데이터로 재조정
                fundsret_mtl = fund_mtl.resample('M').agg(lambda x: (x+1).prod() - 1)

                # 첫 번째 값은 결측값이므로 제거
                fundsret_mtl = fundsret_mtl[1:].iloc[:-2]

                ######
                # Fama-French 5-factor 모델 데이터를 다운로드
                factors = reader.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', LAST_YEAR, TODAY)[0]  # 첫 번째 파트만 가져오기
                factors = factors[1:]  # 첫 번째 행은 결측값이므로 제거

                # 펀드 데이터의 날짜와 Fama-French 데이터의 날짜를 맞춰줌
                fundsret_mtl.index = factors.index

                # 두 데이터를 'Date' 기준으로 병합
                merge = pd.merge(fundsret_mtl, factors, on="Date")

                merge[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']] = merge[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']]/100 #adjust index

                merge['Company_RF'] = merge[funds] - merge['RF']

                y = merge['Company_RF']
                X = merge[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]  # 5개의 요인으로 확장

                X_sm = sm.add_constant(X)

                model = sm.OLS(y,X_sm)
                results = model.fit()

                # 모델 결과 요약을 문자열로 변환
                summary = results.summary().as_text()

                # 주식 데이터의 describe() 결과
                describe_data = data.describe().to_html(classes='table table-striped')

                # 그래프 그리기
                plt.figure(figsize=(10, 6))
                data['Close'].plot(label=f'{funds} - 종가')
                plt.title(f'{funds} 주식 가격 1년치')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()

                # 그래프를 이미지로 저장하기
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                graph_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()

                # 결과를 context로 전달
                context = {
                    'summary': summary,
                    'form': form,
                    'describe_data': describe_data,
                    'graph_data': graph_data
                }
                return render(request, 'registration/predict.html', context)

            except Exception as e:
                error_message = f"알 수 없는 오류가 발생했습니다: {str(e)}"
                return render(request, 'registration/predict.html', {'form': form, 'error': error_message})

    else:
        # 사용자가 처음 페이지를 방문하는 경우 (GET 요청)
        form = TickerForm()

    return render(request, 'registration/predict.html', {'form': form})



