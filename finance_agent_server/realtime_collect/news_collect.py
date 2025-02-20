# -*- coding: utf-8 -*-

# 설치가 필요한 라이브러리
# pip install httpx
# pip install beautifulsoup4
# pip install pytest-playwright
# 반드시 https://playwright.dev/python/docs/intro 에서 Playwright 설치 관련 가이드 참고

# 파이썬 표준 라이브러리
import os
import re
import random
import time
import logging
import traceback
from typing import Callable
from datetime import datetime
from concurrent import futures

# 파이썬 서드파티 라이브러리
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from pymongo import MongoClient

# 사용자 정의 라이브러리
from http_process import sync_request
from preprocessing import *


load_dotenv()


class RealTimeCrawling:
    def __init__(self) -> None:
        self._crawling_scraping = dict()
        self.__possible_websites = ('hankyung', 'maekyung', 'yna')
    
    def add_website(self, website: str) -> bool:
        """크롤링 및 스크래핑할 사이트를 추가하는 메소드
        
        Args:
            website: 크롤링 및 스크래핑할 웹사이트 이름
        
        Returns:
            크롤링 및 스크래핑할 사이트가 성공적으로 추가되었는지 여부, bool
        """

        if website not in self.__possible_websites:
            raise ValueError(f'웹사이트의 이름은 "{", ".join(self.__possible_websites)}" 중 하나여야 합니다.')
        
        self._crawling_scraping[website] = True
        return True
    
    @staticmethod
    def news_crawling(
                        url:str, category: str, note: str, website: str, change_format: str,
                        headers: dict[str, str], log_collection: MongoClient, max_retry: int = 10,
                        min_delay: int | float = 2, max_delay: int | float = 3
                        ) -> dict[str, str | None] | None:
        """뉴스 URL을 바탕으로 크롤링 및 스크래핑을 하는 메소드

        Args:
            url: 뉴스 URL
            category: 뉴스 카테고리
            note: 비고
            website: 웹사이트 이름
            change_format: 바꾸는 포맷
            headers: 식별 정보
            log_collection: MongoDB 로그 log_collection
            max_retry: HTML 문서 정보 불러오기에 실패했을 때 재시도할 최대 횟수
            min_delay: 재시도 할 때 딜레이의 최소 시간
            max_delay: 재시도 할 때 딜레이의 최대 시간

        Returns:
            {
                "news_title": 뉴스 제목, str
                "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                "news_last_upload_time": 뉴스 최종 수정 시각, str | None
                "news_author": 뉴스 작성자, str | None
                "news_content": 뉴스 본문, str
                "news_url": 뉴스 URL, str
                "news_category": 뉴스 카테고리, str
                "news_website": 뉴스 웹사이트, str
                "note": 비고, str | None
            }

            or

            None(크롤링이 실패한 경우)
        """

        info = {} # 뉴스 데이터 정보 Dictionary
        website_dict = {'hankyung': "한국경제", "maekyung": "매일경제", "yna": "연합뉴스"}

        # 동기로 HTML GET
        result = sync_request(url=url, headers=headers, max_retry=max_retry, min_delay=min_delay, max_delay=max_delay)

        # 로그 데이터를 MongoDB에 업로드
        log_dict = {
                        "datetime": datetime.now(), 'website': website, 'note': note, 'category': category, 'url': url,
                    }
        if result['response_status_code'] == 200:
            log_dict['log_level'] = 'info'
        elif result['response_status_code'] != -1:
            log_dict['log_level'] = 'error'
        else:
            log_dict['log_level'] = 'critical'
        log_dict.update({"http_code": result['response_status_code'], 'http_reason': result['response_reason'], 'error_type': result['error_type'], 'error_content': result['error_content'], 'log_content': None})
        log_collection.insert_one(log_dict)

        # HTML 문서 정보를 불러오는 것에 실패하면 None 반환
        if result['html'] is None:
            return None
        # BeautifulSoup로 parser
        soup = BeautifulSoup(result['html'], 'html.parser')

        match website:
            case 'hankyung':
                # 1. 뉴스 데이터의 제목
                title = soup.find('h1', {"class": "headline"})
                title = title.text.strip(' \t\n\r\f\v')

                # 2. 뉴스 데이터의 최초 업로드 시각과 최종 수정 시각
                upload_times = soup.find_all('span', {"class": "txt-date"})

                first_upload_time = upload_times[0].text
                first_upload_time = datetime_trans(website=website, date_time=first_upload_time, change_format=change_format)
                last_upload_time = upload_times[1].text
                last_upload_time = datetime_trans(website=website, date_time=last_upload_time, change_format=change_format)

                # 3. 뉴스 데이터의 기사 작성자
                author_list = soup.find_all('div', {"class": "author link subs_author_list"})
                if author_list:
                    author_list = map(lambda x: x.find("a").text, author_list)
                    author = ', '.join(author_list)
                else:
                    author = None

                # 4. 뉴스 데이터의 본문
                content = soup.find("div", id="articletxt")
                content = content.prettify()
            
            case 'maekyung':
                # 1. 뉴스 데이터의 제목
                title = soup.find("h2", {"class": "news_ttl"})
                title = title.text.strip(' \t\n\r\f\v')

                # 2. 뉴스 데이터의 최초 업로드 시각과 최종 수정 시각
                # 3. 뉴스 데이터의 기사 작성자
                author_datetime = soup.find("div", {"class": "time_info mt-500"})
                all_span = author_datetime.find_all("span")

                authors = []
                first_upload_time = None
                last_upload_time = None

                for s in all_span:
                    s = s.text
                    if '@' in s:
                        s = re.sub(r"\(.+?\)", "", s).strip()
                        authors.append(s)
                    elif '입력' in s:
                        *_, ymd, hms = s.split()
                        first_upload_time = f"{ymd} {hms}"
                        first_upload_time = datetime_trans(website=website, date_time=first_upload_time, change_format=change_format)
                    elif '수정' in s:
                        *_, ymd, hms = s.split()
                        last_upload_time = f"{ymd} {hms}"
                        last_upload_time = datetime_trans(website=website, date_time=last_upload_time, change_format=change_format)

                author = ', '.join(authors)

                # 4. 뉴스 데이터의 본문
                content = soup.find("div", {"class": "news_detail_wrap"})
                content = content.prettify()
            
            case 'yna':
                # 1. 뉴스 데이터의 제목
                title = soup.find("h1", {"class": "tit"})
                title = title.text.strip(' \t\n\r\f\v')

                # 2. 뉴스 데이터의 최초 업로드 시각과 최종 수정 시각
                first_upload_time = soup.find("p", {"class": "update-time"})
                first_upload_time = first_upload_time.text.replace('송고시간', '').strip()
                last_upload_time = None

                # 3. 뉴스 데이터의 기사 작성자
                authors = soup.select('#newsWriterCarousel01 > div > div > div')
                if authors:
                    authors = [f"{author.find('a').text} {author.find('span').text}" for author in authors]
                    author = ', '.join(authors)
                else:
                    author = None

                # 4. 뉴스 데이터의 본문
                content = soup.find('article', {"class": "story-news article"})
                content = content.prettify()
        
        if first_upload_time is not None:
            first_upload_time = datetime.strptime(first_upload_time, change_format)
        if last_upload_time is not None:
            last_upload_time = datetime.strptime(last_upload_time, change_format)

        info['news_title'] = title
        info['news_first_upload_time'] = first_upload_time
        info['news_last_upload_time'] = last_upload_time
        info['news_author'] = author
        info['news_content'] = content
        info['news_url'] = url
        info['news_category'] = category
        info['news_website'] = website_dict[website]
        info['note'] = note

        return info
    
    @staticmethod
    def sync_crawling(
                        url_list: list[str], category: str, note: str, website: str,
                        change_format: str, headers: dict[str, str], news_collection: MongoClient, log_collection: MongoClient,
                        min_delay: int | float = 2, max_delay: int | float = 3
                        ) -> list[dict[str, str | None] | None | int]:
        """동기로 뉴스 URL들을 크롤링 및 스크래핑하는 메소드

        Args:
            url_list: 뉴스 URL list
            category: 뉴스 카테고리
            note: 비고
            website: 웹사이트 이름
            change_format: 바꾸는 포맷
            headers: 식별 정보
            news_collection: MongoDB 뉴스 news_collection
            log_collection: MongoDB 로그 log_collection
            min_delay: 재시도 할 때 딜레이의 최소 시간
            max_delay: 재시도 할 때 딜레이의 최대 시간

        Returns:
            [
                {
                    "news_title": 뉴스 제목, str
                    "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                    "newsfinal_upload_time": 뉴스 최종 수정 시각, str | None
                    "news_author": 뉴스 작성자, str | None
                    "news_content": 뉴스 본문, str
                    "news_url": 뉴스 URL, str
                    "news_category": 뉴스 카테고리, str
                    "news_website": 뉴스 웹사이트, str
                    "note": 비고, str | None
                },
                {
                                        ...
                },
                                        .
                                        .
                                        .
                or

                None(크롤링이 실패한 경우)

                or

                -1(이미 URL이 MongoDB에 존재할 경우)
            ]
        """
        result = []

        if url_list:
            for url in url_list:
                if news_collection.find_one({"news_url": url}) is None:
                    result.append(RealTimeCrawling.news_crawling(url=url, category=category, note=note, website=website, change_format=change_format, headers=headers, log_collection=log_collection, min_delay=min_delay, max_delay=max_delay))
                else:
                    result.append(-1)
                    break
                time.sleep(random.uniform(min_delay, max_delay))
                
        return result
    
    @staticmethod
    def result_summary(
                        url_list: list[str], sync_result: list[dict[str, str | None] | None],
                        website: str, category: str
                    ) -> dict[str, list[dict[str, str | None]] | bool]:
        '''크롤링한 결과를 정리하는 메소드

        Args:
            url_list: 하이퍼링크 리스트
            sync_result: 각 하이퍼링크들로부터 크롤링한 결과 리스트
            website: 웹사이트 이름
            category: 뉴스 카테고리
        
        Returns:
            {
                "result":
                            [
                                {
                                    "news_title": 뉴스 제목, str
                                    "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                                    "newsfinal_upload_time": 뉴스 최종 수정 시각, str | None
                                    "news_author": 뉴스 작성자, str | None
                                    "news_content": 뉴스 본문, str
                                    "news_url": 뉴스 URL, str
                                    "news_category": 뉴스 카테고리, str
                                    "news_website": 뉴스 웹사이트, str
                                    "note": 비고, str | None
                                },
                                {
                                                        ...
                                },
                                                        .
                                                        .
                                                        .
                            ],
                "nonstop": bool
            }
        '''

        info = {"result": [], "nonstop": True}

        # 요청이 실패했으면 제외
        for res in sync_result:
            if res is None:
                pass
            elif res == -1:
                info["nonstop"] = False
                break
            else:
                info['result'].append(res)

        return info

    @staticmethod
    def multithread_category(
                                website: str, website_category_function: Callable[[str, dict[str, str], MongoClient, str, int | float, int | float], list[dict[str, str | None]]],
                                category_list: list[str], change_format: str, headers: dict[str, str],
                                news_collection: MongoClient, log_collection: MongoClient,
                                min_delay: int | float = 2, max_delay: int | float = 3
                             ) -> list[dict[str, str | None]]:
        '''멀티 스레딩으로 웹사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            website: 웹사이트 이름
            website_category_function: 각 웹사이트별 category 크롤링 함수
            category_list: category 리스트
            change_format: 바꾸는 포맷
            headers: 식별 정보
            news_collection: MongoDB 뉴스 news_collection
            log_collection: MongoDB 로그 log_collection
            min_delay: 재시도 할 때 딜레이의 최소 시간
            max_delay: 재시도 할 때 딜레이의 최대 시간

        Returns:
            [
                {
                    "news_title": 뉴스 제목, str
                    "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                    "newsfinal_upload_time": 뉴스 최종 수정 시각, str | None
                    "news_author": 뉴스 작성자, str | None
                    "news_content": 뉴스 본문, str
                    "news_url": 뉴스 URL, str
                    "news_category": 뉴스 카테고리, str
                    "news_website": 뉴스 웹사이트, str
                    "note": 비고, str | None
                },
                {
                                        ...
                },
                                        .
                                        .
                                        .
            ]
        '''

        n = len(category_list)
        website_result = []
        
        with futures.ThreadPoolExecutor(max_workers=n) as executor:
            future_to_category = {executor.submit(website_category_function, change_format, headers, news_collection, log_collection, category, min_delay, max_delay): category for category in category_list}
            for future in futures.as_completed(future_to_category):
                category = future_to_category[future]
                try:
                    website_result.extend(future.result())
                except Exception as e:
                    # 로그 데이터를 MongoDB에 업로드
                    log_collection.insert_one({"date": datetime.now(), "website": website, 'note': None, 'category': category, 'url': None, 'log_level': "critical", "http_code": None, "http_reason": None, "error_type": type(e).__name__, "error_content": traceback.format_exc(), "log_content": None})
        # 로그 데이터를 MongoDB에 업로드
        log_collection.insert_one({"date": datetime.now(), "website": website, 'note': None, 'category': None, 'url': None, 'log_level': "debug", "http_code": None, "http_reason": None, "error_type": None, "error_content": None, "log_content": f"{website} finish"})
        return website_result
    
    @staticmethod
    def hankyung(
                change_format: str, headers: dict[str, str],
                min_delay: int | float = 2, max_delay: int | float = 5.5
                ) -> list[dict[str, str | None]]:
        """hankyung 사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            change_format: 바꾸는 포맷
            headers: 식별 정보
            min_delay: 재시도 할 때 딜레이의 최소 시간
            max_delay: 재시도 할 때 딜레이의 최대 시간

        Returns:
            [
                {
                    "news_title": 뉴스 제목, str
                    "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                    "newsfinal_upload_time": 뉴스 최종 수정 시각, str | None
                    "news_author": 뉴스 작성자, str | None
                    "news_content": 뉴스 본문, str
                    "news_url": 뉴스 URL, str
                    "news_category": 뉴스 카테고리, str
                    "news_website": 뉴스 웹사이트, str
                    "note": 비고, str | None
                },
                {
                                        ...
                },
                                        .
                                        .
                                        .
            ]
        """

        def hankyung_url(category: str, page: int, industry_set: set[str], globalmarket_set: set[str]) -> str:
            """hankyung 사이트에서 어떤 카테고리인지에 따라 해당 page의 url을 반환하는 함수
            
            Args:
                category: 뉴스 카테고리
                page: 페이지
                industry_set: 산업 집합
                globalmarket_set: 글로벌 마켓 집합

            Returns:
                hankyung 사이트에서 해당 카테고리의 해당 페이지 URL
            """

            if category in industry_set:
                return f'https://www.hankyung.com/industry/{category}?page={page}'
            elif category in globalmarket_set:
                return f'https://www.hankyung.com/globalmarket/{category}?page={page}'

        def hankyung_category(
                                change_format: str, headers: dict[str, str], news_collection: MongoClient, log_collection: MongoClient,
                                category: str, min_delay: int | float = 2, max_delay: int | float = 3
                            ) -> list[dict[str, str | None]]:
            """hankyung 사이트를 크롤링 및 스크래핑 하는 메소드

            Args:
                change_format: 바꾸는 포맷
                headers: 식별 정보
                news_collection: MongoDB 뉴스 news_collection
                log_collection: MongoDB 로그 log_collection
                category: 뉴스 카테고리
                min_delay: 재시도 할 때 딜레이의 최소 시간
                max_delay: 재시도 할 때 딜레이의 최대 시간

            Returns:
                [
                    {
                        "news_title": 뉴스 제목, str
                        "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                        "newsfinal_upload_time": 뉴스 최종 수정 시각, str | None
                        "news_author": 뉴스 작성자, str | None
                        "news_content": 뉴스 본문, str
                        "news_url": 뉴스 URL, str
                        "news_category": 뉴스 카테고리, str
                        "news_website": 뉴스 웹사이트, str
                        "note": 비고, str | None
                    },
                    {
                                            ...
                    },
                                            .
                                            .
                                            .
                ]
            """

            category_dict = {
                                "semicon-electronics": "반도체·전자", "auto-battery": "자동차·배터리", "ship-marine": "조선·해운",
                                "steel-chemical": "철강·화학", "robot-future": "로봇·미래", "manage-business": "경영·재계",
                                "build-machinery": "건설·기계", "news-wallstreet-now": "김현석의 월스트리트나우", "news-market": "마켓 트렌드",
                                "news-stock-focus": "종목 포커스", "news-global-stock": "글로벌 종목탐구", "news-portfolio": "대가들의 포트폴리오",
                                "news-global-etf": "글로벌 ETF 트렌드", "news-global-macro": "글로벌 매크로", "news-global-issue": "글로벌 이슈",
                                "news-global-report": "글로벌 리포트", "news-fed": "Fed 워치", "news-wti": "오늘의 유가",
                                "news-commodities-indices": "원자재 포커스", "news-forex": "외환", "news-bond": "채권"              
                            }
            industry_set = set((
                                    'semicon-electronics', 'auto-battery', 'ship-marine', 'steel-chemical',
                                    'robot-future', 'manage-business', 'build-machinery'
                                ))
            globalmarket_set = set((
                                        "news-wallstreet-now", "news-market",
                                        "news-stock-focus", "news-global-stock", "news-portfolio",
                                        "news-global-etf", "news-global-macro", "news-global-issue",
                                        "news-global-report", "news-fed", "news-wti",
                                        "news-commodities-indices", "news-forex", "news-bond"
                                    ))

            page = 1
            website = 'hankyung'
            nonstop = True
            if category in industry_set:
                note = '산업'
            elif category in globalmarket_set:
                note = '글로벌 마켓'
            hankyung_results = []

            while nonstop:
                time.sleep(random.uniform(min_delay, max_delay))
                web_page = hankyung_url(category=category, page=page, industry_set=industry_set, globalmarket_set=globalmarket_set)
                sync_result = sync_request(url=web_page, headers=headers)
                
                # html 문서 불러오기에 실패했으면 다음 페이지로 넘기기
                if sync_result['html'] is None:
                    # 로그 데이터를 MongoDB에 업로드
                    log_dict = {
                                    "datetime": datetime.now(), 'website': website, 'note': note, 'category': category, 'url': web_page,
                                }
                    if sync_result['response_status_code'] == 200:
                        log_dict['log_level'] = 'info'
                    elif sync_result['response_status_code'] != -1:
                        log_dict['log_level'] = 'error'
                    else:
                        log_dict['log_level'] = 'critical'
                    log_dict.update({"http_code": sync_result['response_status_code'], 'http_reason': sync_result['response_reason'], 'error_type': sync_result['error_type'], 'error_content': sync_result['error_content'], 'log_content': None})
                    log_collection.insert_one(log_dict)
                    continue

                soup = BeautifulSoup(sync_result['html'], 'html.parser')
                url_tag_list = soup.find_all('h2', {"class": "news-tit"})

                # url_tag_list가 비어있으면 최종 페이지까지 갔다는 것이므로 종료
                if not url_tag_list:
                    nonstop = False
                    break

                url_list = [url_tag.find('a')["href"] for url_tag in url_tag_list]

                sync_result = RealTimeCrawling.sync_crawling(url_list=url_list, category=category_dict[category], note=note, website=website, change_format=change_format, headers=headers, news_collection=news_collection, log_collection=log_collection)
                
                # 결과 정리
                info = RealTimeCrawling.result_summary(url_list=url_list, sync_result=sync_result, website=website, category=category)
                result, nonstop = info['result'], info['nonstop']

                # 본문 전처리
                hankyung_preprocess(result)

                hankyung_results.extend(result)
                log_collection.insert_one({"date": datetime.now(), "website": website, 'note': note, 'category': category_dict[category], 'url': web_page, 'log_level': "info", "http_code": None, "http_reason": None, "error_type": None, "error_content": None, "log_content": "in progress"})
                page += 1
            log_collection.insert_one({"date": datetime.now(), "website": website, 'note': note, 'category': category_dict[category], 'url': web_page, 'log_level': "debug", "http_code": None, "http_reason": None, "error_type": None, "error_content": None, "log_content": f"{category} finish"})
            return hankyung_results
        
        cluster = MongoClient(os.getenv("mongo")) # 클러스터
        news_db = cluster['Document'] # 뉴스 정보
        news_collection = news_db["newsdata"] # 내가 지정한 컬렉션 이름
        log_db = cluster['log']
        log_collection = log_db['crawling']

        category_list = [
                            "semicon-electronics", "auto-battery", "ship-marine",
                            "steel-chemical", "robot-future", "manage-business",
                            "build-machinery", "news-wallstreet-now", "news-market",
                            "news-stock-focus", "news-global-stock", "news-portfolio",
                            "news-global-etf", "news-global-macro", "news-global-issue",
                            "news-global-report", "news-fed", "news-wti",
                            "news-commodities-indices", "news-forex", "news-bond"    
                        ]
        
        hankyung_result = RealTimeCrawling.multithread_category(website='hankyung', website_category_function=hankyung_category, category_list=category_list, change_format=change_format, headers=headers, news_collection=news_collection, log_collection=log_collection,  min_delay=min_delay, max_delay=max_delay)
        news_collection.insert_many(hankyung_result)
        cluster.close()
        return hankyung_result

    @staticmethod
    def maekyung(
                change_format: str, headers: dict[str, str],
                min_delay: int | float = 2, max_delay: int | float = 3
                ) -> list[dict[str, str | None]]:
        """maekyung 사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            change_format: 바꾸는 포맷
            headers: 식별 정보
            min_delay: 재시도 할 때 딜레이의 최소 시간
            max_delay: 재시도 할 때 딜레이의 최대 시간

        Returns:
            [
                {
                    "news_title": 뉴스 제목, str
                    "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                    "newsfinal_upload_time": 뉴스 최종 수정 시각, str | None
                    "news_author": 뉴스 작성자, str | None
                    "news_content": 뉴스 본문, str
                    "news_url": 뉴스 URL, str
                    "news_category": 뉴스 카테고리, str
                    "news_website": 뉴스 웹사이트, str
                    "note": 비고, str | None
                },
                {
                                        ...
                },
                                        .
                                        .
                                        .
            ]
        """

        def maekyung_url(category: str, page: int) -> str:
            """maekyung 사이트에서 어떤 카테고리인지에 따라 해당 page의 url을 반환하는 함수
            
            Args:
                category: 뉴스 카테고리
                page: 페이지

            Returns:
                markyung 사이트에서 해당 카테고리의 해당 페이지 URL
            """

            return f"https://stock.mk.co.kr/news/{category}?page={page}"
        
        def maekyung_category(
                                change_format: str, headers: dict[str, str], news_collection: MongoClient, log_collection: MongoClient,
                                category: str, min_delay: int | float = 2, max_delay: int | float = 3
                            ) -> list[dict[str, str | None]]:
            """maekyung 사이트를 크롤링 및 스크래핑 하는 메소드

            Args:
                change_format: 바꾸는 포맷
                headers: 식별 정보
                news_collection: MongoDB 뉴스 news_collection
                log_collection: MongoDB 로그 log_collection
                category: 뉴스 카테고리
                min_delay: 재시도 할 때 딜레이의 최소 시간
                max_delay: 재시도 할 때 딜레이의 최대 시간

            Returns:
                [
                    {
                        "news_title": 뉴스 제목, str
                        "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                        "newsfinal_upload_time": 뉴스 최종 수정 시각, str | None
                        "news_author": 뉴스 작성자, str | None
                        "news_content": 뉴스 본문, str
                        "news_url": 뉴스 URL, str
                        "news_category": 뉴스 카테고리, str
                        "news_website": 뉴스 웹사이트, str
                        "note": 비고, str | None
                    },
                    {
                                            ...
                    },
                                            .
                                            .
                                            .
                ]
            """

            category_dict = {
                                "marketCondition": "증권정책·시황", "company": "종목분석·기업정보", "world": "해외증시",
                                "bond": "채권·펀드·선물·옵션"       
                            }

            page = 1
            website = 'maekyung'
            nonstop = True
            note = "마켓"
            maekyung_results = []

            while nonstop:
                time.sleep(random.uniform(min_delay, max_delay))
                web_page = maekyung_url(category=category, page=page)
                sync_result = sync_request(url=web_page, headers=headers)
                
                # html 문서 불러오기에 실패했으면 다음 페이지로 넘기기
                if sync_result['html'] is None:
                    # 로그 데이터를 MongoDB에 업로드
                    log_dict = {
                                    "datetime": datetime.now(), 'website': website, 'note': note, 'category': category, 'url': web_page,
                                }
                    if sync_result['response_status_code'] == 200:
                        log_dict['log_level'] = 'info'
                    elif sync_result['response_status_code'] != -1:
                        log_dict['log_level'] = 'error'
                    else:
                        log_dict['log_level'] = 'critical'
                    log_dict.update({"http_code": sync_result['response_status_code'], 'http_reason': sync_result['response_reason'], 'error_type': sync_result['error_type'], 'error_content': sync_result['error_content'], 'log_content': None})
                    log_collection.insert_one(log_dict)
                    continue

                soup = BeautifulSoup(sync_result['html'], 'html.parser')
                news_container = soup.find("section", {"class": "news_sec latest_news_sec"})
                news_list = news_container.find_all("li", {"class": "news_node"})
                url_tag_list = [news.find("a", {"class": "news_item"})["href"] for news in news_list]

                # url_tag_list가 비어있으면 최종 페이지까지 갔다는 것이므로 종료
                if not url_tag_list:
                    nonstop = False
                    break

                url_list = [f"https://stock.mk.co.kr{url}" for url in url_tag_list]

                sync_result = RealTimeCrawling.sync_crawling(url_list=url_list, category=category_dict[category], note=note, website=website, change_format=change_format, headers=headers, news_collection=news_collection, log_collection=log_collection)
                
                # 결과 정리
                info = RealTimeCrawling.result_summary(url_list=url_list, sync_result=sync_result, website=website, category=category)
                result, nonstop = info['result'], info['nonstop']

                # 본문 전처리
                maekyung_preprocess(result)

                maekyung_results.extend(result)
                log_collection.insert_one({"date": datetime.now(), "website": website, 'note': note, 'category': category_dict[category], 'url': web_page, 'log_level': "info", "http_code": None, "http_reason": None, "error_type": None, "error_content": None, "log_content": "in progress"})
                page += 1
            log_collection.insert_one({"date": datetime.now(), "website": website, 'note': note, 'category': category_dict[category], 'url': web_page, 'log_level': "debug", "http_code": None, "http_reason": None, "error_type": None, "error_content": None, "log_content": f"{category} finish"})
            return maekyung_results
        
        cluster = MongoClient(os.getenv("mongo")) # 클러스터
        news_db = cluster['Document'] # 뉴스 정보
        news_collection = news_db["newsdata"] # 내가 지정한 컬렉션 이름
        log_db = cluster['log']
        log_collection = log_db['crawling']

        category_list = [
                            'company'
                        ]

        maekyung_result = RealTimeCrawling.multithread_category(website='maekyung', website_category_function=maekyung_category, category_list=category_list, change_format=change_format, headers=headers, news_collection=news_collection, log_collection=log_collection, min_delay=min_delay, max_delay=max_delay)
        news_collection.insert_many(maekyung_result)
        cluster.close()
        return maekyung_result

    @staticmethod
    def yna(
                change_format: str, headers: dict[str, str],
                min_delay: int | float = 2, max_delay: int | float = 3
            ) -> list[dict[str, str | None]]:
        """yna 사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            change_format: 바꾸는 포맷
            headers: 식별 정보
            min_delay: 재시도 할 때 딜레이의 최소 시간
            max_delay: 재시도 할 때 딜레이의 최대 시간

        Returns:
            [
                {
                    "news_title": 뉴스 제목, str
                    "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                    "newsfinal_upload_time": 뉴스 최종 수정 시각, str | None
                    "news_author": 뉴스 작성자, str | None
                    "news_content": 뉴스 본문, str
                    "news_url": 뉴스 URL, str
                    "news_category": 뉴스 카테고리, str
                    "news_website": 뉴스 웹사이트, str
                    "note": 비고, str | None
                },
                {
                                        ...
                },
                                        .
                                        .
                                        .
            ]
        """

        def yna_url(category: str, page: int, marketplus_set: set[str]) -> str:
            """yna사이트에서 어떤 카테고리인지에 따라 해당 page의 url을 반환하는 함수
            
            Args:
                category: 뉴스 카테고리
                page: 페이지
                marketplus_set: 마켓 플러스 집합

            Returns:
                yna 사이트에서 해당 카테고리의 해당 페이지 URL
            """

            if category in marketplus_set:
                return f'https://www.yna.co.kr/market-plus/{category}/{page}'

        def yna_category(
                            change_format: str, headers: dict[str, str], news_collection: MongoClient, log_collection: MongoClient,
                            category: str, min_delay: int | float = 2, max_delay: int | float = 3
                        ) -> list[dict[str, str | None]]:
            """yna 사이트를 크롤링 및 스크래핑 하는 메소드

            Args:
                change_format: 바꾸는 포맷
                headers: 식별 정보
                news_collection: MongoDB 뉴스 news_collection
                log_collection: MongoDB 로그 log_collection
                category: 뉴스 카테고리
                min_delay: 재시도 할 때 딜레이의 최소 시간
                max_delay: 재시도 할 때 딜레이의 최대 시간

            Returns:
                [
                    {
                        "news_title": 뉴스 제목, str
                        "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                        "newsfinal_upload_time": 뉴스 최종 수정 시각, str | None
                        "news_author": 뉴스 작성자, str | None
                        "news_content": 뉴스 본문, str
                        "news_url": 뉴스 URL, str
                        "news_category": 뉴스 카테고리, str
                        "news_website": 뉴스 웹사이트, str
                        "note": 비고, str | None
                    },
                    {
                                            ...
                    },
                                            .
                                            .
                                            .
                ]
            """

            category_dict = {
                                "domestic-stock": "국내주식", "global-stock": "해외주식", "bond": "채권",
                                "fund-etf": "펀드/ETF", "global-market": "글로벌시장", "financial-investment": "증권/운용사",
                                "report": "리포트", "disclosure": "공시"
                            }
            marketplus_set = set((
                                    "domestic-stock", "global-stock", "bond",
                                    "fund-etf", "global-market", "financial-investment",
                                    "report", "disclosure"
                                    ))

            page = 1
            website = 'yna'
            nonstop = True
            if category in marketplus_set:
                note = '마켓+'
            yna_results = []

            with sync_playwright() as p:
                # 브라우저 열기(Chromium, Firefox, WebKit 중 하나 선택 가능)
                browser = p.chromium.launch(headless=True)
                webpage = browser.new_page()

                while nonstop:
                    # 웹사이트로 이동
                    time.sleep(random.uniform(min_delay, max_delay))
                    web_page = yna_url(category=category, page=page, marketplus_set=marketplus_set)
                    webpage.goto(web_page)

                    news_container = webpage.locator('//*[@id="container"]/div/div/div[2]/section/div[1]/ul')
                    news_list = news_container.locator('//*[@class="news-con"]')
                    news_cnt = news_list.count()

                    # 뉴스기사가 없으면 종료
                    if news_cnt == 0:
                        nonstop = False
                        break

                    news_list = news_list.all()
                    url_list = [news.locator("a").get_attribute("href") for news in news_list]

                    sync_result = RealTimeCrawling.sync_crawling(url_list=url_list, category=category_dict[category], note=note, website=website, change_format=change_format, headers=headers, news_collection=news_collection, log_collection=log_collection)
                
                    # 결과 정리
                    info = RealTimeCrawling.result_summary(url_list=url_list, sync_result=sync_result, website=website, category=category)
                    result, nonstop = info['result'], info['nonstop']

                    # 본문 전처리
                    yna_preprocess(result)

                    yna_results.extend(result)
                    log_collection.insert_one({"date": datetime.now(), "website": website, 'note': note, 'category': category_dict[category], 'url': web_page, 'log_level': "info", "http_code": None, "http_reason": None, "error_type": None, "error_content": None, "log_content": "in progress"})
                    page += 1
                # 작업 후 브라우저 닫기
                browser.close()
            log_collection.insert_one({"date": datetime.now(), "website": website, 'note': note, 'category': category_dict[category], 'url': web_page, 'log_level': "debug", "http_code": None, "http_reason": None, "error_type": None, "error_content": None, "log_content": f"{category} finish"})
            return yna_results
        
        cluster = MongoClient(os.getenv("mongo")) # 클러스터
        news_db = cluster['Document'] # 뉴스 정보
        news_collection = news_db["newsdata"] # 내가 지정한 컬렉션 이름
        log_db = cluster['log']
        log_collection = log_db['crawling']
        
        category_list = [
                            'global-stock', 'global-market'
                        ]

        yna_result = RealTimeCrawling.multithread_category(website='yna', website_category_function=yna_category, category_list=category_list, change_format=change_format, headers=headers, news_collection=news_collection, log_collection=log_collection, min_delay=min_delay, max_delay=max_delay)
        news_collection.insert_many(yna_result)
        cluster.close()
        return yna_result
    
    @staticmethod
    def web_crawling(
                    website: str, change_format: str
                    ) -> list[dict[str, str | None]]:
        """해당 웹사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            website: 웹사이트 이름
            change_format: 바꾸는 포맷

        Returns:
            [
                {
                    "news_title": 뉴스 제목, str
                    "news_first_upload_time": 뉴스 최초 업로드 시각, str | None
                    "newsfinal_upload_time": 뉴스 최종 수정 시각, str | None
                    "news_author": 뉴스 작성자, str | None
                    "news_content": 뉴스 본문, str
                    "news_url": 뉴스 URL, str
                    "news_category": 뉴스 카테고리, str
                    "news_website": 뉴스 웹사이트, str
                    "note": 비고, str | None
                },
                {
                                        ...
                },
                                        .
                                        .
                                        .
            ]
        """

        # User-Agent 변경을 위한 옵션 설정
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        headers = {
                    'User-Agent': user_agent,
                    "Connection": "close"
                }

        match website:
            case 'hankyung':
                return RealTimeCrawling.hankyung(change_format=change_format, headers=headers, min_delay=3.5, max_delay=10)
            case 'maekyung':
                return RealTimeCrawling.maekyung(change_format=change_format, headers=headers, min_delay=3.5, max_delay=10)
            case 'yna':
                return RealTimeCrawling.yna(change_format=change_format, headers=headers, min_delay=3.5, max_delay=10)

    def run(self, change_format: str = '%Y-%m-%d %H:%M') -> None:
        """멀티 프로세싱 및 멀티 스레딩으로 웹사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            change_format: 바꾸는 포맷

        Returns:
            None
        """

        website_list = list(self._crawling_scraping.keys())
        n = len(self._crawling_scraping)

        with futures.ProcessPoolExecutor(max_workers=n) as executor:
            future_to_website = {executor.submit(RealTimeCrawling.web_crawling, website, change_format): website for website in website_list}
            for future in futures.as_completed(future_to_website):
                website = future_to_website[future]

        return None