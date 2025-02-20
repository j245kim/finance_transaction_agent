import yfinance as yf
import pandas as pd
import requests

def get_sp500_return(start_date="2023-01-01", end_date="2024-01-01"):
    """
    S&P 500 (^GSPC) 지수의 수익률을 계산하는 함수

    :param start_date: 조회 시작 날짜 (YYYY-MM-DD)
    :param end_date: 조회 종료 날짜 (YYYY-MM-DD)
    :return: S&P 500 수익률, 시작 가격, 종료 가격
    """
    # S&P 500 티커 (^GSPC) 데이터 다운로드
    sp500 = yf.Ticker("^GSPC")
    data = sp500.history(start=start_date, end=end_date)

    if data.empty:
        print(f"Error: No S&P 500 data found for {start_date} to {end_date}")
        return None, None, None

    # 시작 가격과 종료 가격 가져오기
    start_sp500_price = data["Close"].iloc[0]  # 첫날 종가
    end_sp500_price = data["Close"].iloc[-1]   # 마지막날 종가

    # 수익률 계산 (%)
    return_rate = ((end_sp500_price - start_sp500_price) / start_sp500_price) * 100

    return return_rate, start_sp500_price, end_sp500_price


def get_gold_price(date=""):
    """ 특정 날짜의 금 시세를 조회하는 함수 """
    api_key = "goldapi-2dxu3lsm6y7hdk6-io"
    symbol = "XAU"
    curr = "USD"
    
    url = f"https://www.goldapi.io/api/{symbol}/{curr}/{date}"
    
    headers = {
        "x-access-token": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("price", None)  # 현재 온스당 금 가격
    except requests.exceptions.RequestException as e:
        print("Error:", str(e))
        return None


def compare_gold_prices(start_year, end_year):
    """
    기준년도와 비교년도 금 시세 및 S&P 500 지수 및 수익률을 계산하는 함수.
    
    :param start_year: 비교 기준 년도 (예: "2023")
    :param end_year: 비교 대상 년도 (예: "2024")
    :return: 금 & S&P 500 가격 및 수익률 정보 딕셔너리 반환
    """
    # 날짜 변환
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"

    # 📌 금 가격 데이터 가져오기
    start_gold_price = get_gold_price(start_date)
    end_gold_price = get_gold_price(end_date)

    # 📌 S&P 500 데이터 가져오기
    sp500_return, start_sp500_price, end_sp500_price = get_sp500_return(start_date, end_date)

    # 📌 금 수익률 계산
    gold_return = ((end_gold_price - start_gold_price) / start_gold_price * 100) if start_gold_price and end_gold_price else None

    return {
        # "기준년도 금 가격": start_gold_price,
        # "비교년도 금 가격": end_gold_price,
        # "기준년도 S&P 500 가격": start_sp500_price,
        # "비교년도 S&P 500 가격": end_sp500_price,
        "금 수익률 (%)": round(gold_return, 2) if gold_return is not None else "데이터 없음",
        "S&P 500 수익률 (%)": round(sp500_return, 2) if sp500_return is not None else "데이터 없음"
    }


# # 실행 예제
# result = compare_gold_prices("2023", "2024")
# for key, value in result.items():
#     print(f"{key}: {value}")
