# -*- coding: utf-8 -*-

# 파이썬 표준 라이브러리
import re
from datetime import datetime
from copy import deepcopy


def datetime_trans(website: str, date_time: str, change_format: str) -> str:
    """웹사이트에 따른 업로드 시각, 수정 시각들을 같은 포맷으로 바꾸는 함수

    Args:
        website: 웹사이트 이름
        date_time: 바꿀 시각
        change_format: 바꾸는 포맷
    
    Return:
        2000-01-01 23:59 형태등의 str, 단 포맷은 사용자가 지정 가능
    """
        
    match website:
        case 'hankyung':
            news_datetime = date_time.replace('.', '-')
            news_datetime = datetime.strptime(news_datetime, '%Y-%m-%d %H:%M')
        case 'maekyung':
            news_datetime = date_time.replace('.', '-')
            news_datetime = datetime.strptime(news_datetime, '%Y-%m-%d %H:%M:%S')
        case 'yna':
            news_datetime = datetime.strptime(news_datetime, '%Y-%m-%d %H:%M')

    news_datetime = datetime.strftime(news_datetime, change_format)

    return news_datetime


def datetime_cut(
                news_list: list[dict[str, str, None]],
                end_date: datetime, change_format: str
                ) -> dict[str, list[dict[str, str, None]], bool]:
    """end_date보다 빠른 날짜의 데이터들을 제거하는 함수

    Args:
        news_list: 크롤링 및 스크래핑한 뉴스 데이터들
        end_date: 기준 시각
        change_format: 바꾸는 포맷
    
    Returns
        {
            "result": 자르기 완료한 크롤링 및 스크래핑 뉴스 데이터들, list[dict[str, str, None]]
            "nonstop": 진행 여부 부울 변수, bool
        }
    """
    
    info = {"result": deepcopy(news_list), 'nonstop': True}

    if not news_list:
         info['nonstop'] = False

    while info['result'] and (datetime.strptime(info['result'][-1]['news_first_upload_time'], change_format) < end_date):
            info['nonstop'] = False
            del info['result'][-1]

    return info