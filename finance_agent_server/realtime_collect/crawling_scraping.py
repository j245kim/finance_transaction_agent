# -*- coding: utf-8 -*-

# 설치가 필요한 라이브러리
# pip install httpx
# pip install beautifulsoup4
# pip install pytest-playwright
# 반드시 https://playwright.dev/python/docs/intro 에서 Playwright 설치 관련 가이드 참고

# 파이썬 표준 라이브러리
import os
import json
import re
import random
import time
import asyncio
import logging
import traceback
from datetime import datetime
from functools import partial
from concurrent import futures
from pathlib import Path

# 파이썬 서드파티 라이브러리
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# 사용자 정의 라이브러리
from .http_process import sync_request, async_request
from .preprocessing import datetime_trans, datetime_cut


class NewsInfo:
    def __init__(self, website: str, save_path: str) -> None:
        self._results = [] # 크롤링 및 스크래핑한 데이터들
        self.__website = website # 크롤링 및 스크래핑한 사이트 이름
        self.__save_path = save_path # 저장 경로

    def __len__(self) -> int:
        return len(self._results)
    
    def __eq__(self, other) -> bool:
        return self._results == other
    
    def to_json(self) -> None:
        """크롤링 및 스크래핑한 데이터를 json 파일로 저장하는 메소드"""

        with open(self.__save_path, mode='w', encoding='utf-8') as f:
            json.dump(self._results, f, ensure_ascii=False, indent=4)

class CrawlingScraping:
    def __init__(self, record_log: bool = False) -> None:
        self._crawling_scraping = dict()
        self.__possible_websites = ('hankyung', 'maekyung', 'yna')
        self.__stfo_path = Path(__file__).parents[1]
        self.__logs_path = rf'{self.__stfo_path}\logs'
        self.__logs_data_path = rf'{self.__stfo_path}\logs\crawling_scraping_log'
        self.__datas_path = rf'{self.__stfo_path}\datas'
        self.__datas_news_path = rf'{self.__stfo_path}\datas\news_data'

        if record_log:
            # logs 폴더가 없으면 logs 폴더와 그 하위 폴더로 crawling_scraping_log 폴더 생성
            if not os.path.exists(self.__logs_path):
                os.makedirs(self.__logs_data_path, exist_ok=True)
            # logs 폴더는 있지만 crawling_scraping_log 폴더가 없으면 crawling_scraping_log 폴더 생성
            elif not os.path.exists(self.__logs_data_path):
                os.mkdir(self.__logs_data_path)
        # datas 폴더가 없으면 datas 폴더와 그 하위 폴더로 news_data 폴더 생성
        if not os.path.exists(self.__datas_path):
            os.makedirs(self.__datas_news_path, exist_ok=True)
        # datas 폴더는 있지만 newsdata 폴더가 없으면 news_data 폴더 생성
        elif not os.path.exists(self.__datas_news_path):
            os.mkdir(self.__datas_news_path)
    
    def add_website(self, website: str, save_path: str = None) -> bool:
        """크롤링 및 스크래핑할 사이트를 추가하는 메소드
        
        Args:
            website: 크롤링 및 스크래핑할 웹사이트 이름
            save_path: 크롤링 및 스크래핑한 데이터를 저장할 경로
        
        Returns:
            크롤링 및 스크래핑할 사이트가 성공적으로 추가되었는지 여부, bool
        """

        if website not in self.__possible_websites:
            raise ValueError(f'웹사이트의 이름은 "{", ".join(self.__possible_websites)}" 중 하나여야 합니다.')
        
        if save_path is None:
            save_path = rf'{self.__datas_news_path}\{website}_data.json'
        self._crawling_scraping[website] = NewsInfo(website=website, save_path=save_path)
        return True
    
    @staticmethod
    async def news_crawling(
                            url:str, category: str, note: str, website: str, change_format: str,
                            headers: dict[str, str], max_retry: int = 10,
                            min_delay: int | float = 2, max_delay: int | float = 3
                            ) -> dict[str, str, None] | None:
        """뉴스 URL을 바탕으로 크롤링 및 스크래핑을 하는 메소드

        Args:
            url: 뉴스 URL
            category: 뉴스 카테고리
            note: 비고
            website: 웹사이트 이름
            change_format: 바꾸는 포맷
            headers: 식별 정보
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

            None
        """

        info = {} # 뉴스 데이터 정보 Dictionary
        website_dict = {'hankyung': "한국경제", "maekyung": "매일경제", "yna": "연합뉴스"}

        # 비동기로 HTML GET
        result = await async_request(url=url, headers=headers, max_retry=max_retry, min_delay=min_delay, max_delay=max_delay)
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
    async def async_crawling(
                            url_list: list[str], category: str, note: str, website: str,
                            change_format: str, headers: dict[str, str],
                            min_delay: int | float = 2, max_delay: int | float = 3
                            ) -> list[dict[str, str, None], None]:
        """비동기로 뉴스 URL들을 크롤링 및 스크래핑하는 메소드

        Args:
            url_list: 뉴스 URL list
            category: 뉴스 카테고리
            note: 비고
            website: 웹사이트 이름
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

        if url_list:
            crawl_list = [CrawlingScraping.news_crawling(url=url, category=category, note=note, website=website, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay) for url in url_list]
            async_result = await asyncio.gather(*crawl_list)
            return async_result
        return []
    
    @staticmethod
    async def hankyung(
                        end_datetime: str, date_format: str,
                        change_format: str, headers: dict[str, str],
                        min_delay: int | float = 2, max_delay: int | float = 5.5
                        ) -> list[dict[str, str]]:
        """hankyung 사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            end_datetime: 크롤링 및 스크래핑할 마지막 시각
            date_format: 시각 포맷
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

        async def hankyung_category(
                                        end_datetime: str, date_format: str,
                                        change_format: str, headers: dict[str, str], category: str,
                                        min_delay: int | float = 2, max_delay: int | float = 3
                                    ) -> list[dict[str, str, None]]:
            """hankyung 사이트를 크롤링 및 스크래핑 하는 메소드

            Args:
                end_datetime: 크롤링 및 스크래핑할 마지막 시각
                date_format: 시각 포맷
                change_format: 바꾸는 포맷
                headers: 식별 정보
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
            end_date = datetime.strptime(end_datetime, date_format)
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
                    print()
                    print(f'{page}번 페이지의 HTML 문서 정보를 불러오는데 실패했습니다.')
                    continue

                soup = BeautifulSoup(sync_result['html'], 'html.parser')
                url_tag_list = soup.find_all('h2', {"class": "news-tit"})

                # url_tag_list가 비어있으면 최종 페이지까지 갔다는 것이므로 종료
                if not url_tag_list:
                    nonstop = False
                    break

                url_list = [url_tag.find('a')["href"] for url_tag in url_tag_list]

                async_result = await CrawlingScraping.async_crawling(url_list=url_list, category=category_dict[category], note=note, website=website, change_format=change_format, headers=headers)
                
                # 요청이 실패했으면 제외
                result = []
                for idx, res in enumerate(async_result):
                    if res is None:
                        print()
                        print(f'요청 실패한 데이터 : URL={url_list[idx]}, category={category}, website={website}')
                    else:
                        result.append(res)
                
                # end_date 이후가 아니면은 제거
                cut_info = datetime_cut(news_list=result, end_date=end_date, change_format=change_format)
                result, nonstop = cut_info['result'], cut_info['nonstop']

                hankyung_results.extend(result)
                page += 1
                print(f'website: {website}, category: {category_dict[category]}, page: {page}')
            print(f'website: {website}, category: {category_dict[category]} 종료')
            return hankyung_results
        
        async with asyncio.TaskGroup() as tg:
            semicon_electronics = tg.create_task(hankyung_category(category='semicon-electronics', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            auto_battery = tg.create_task(hankyung_category(category='auto-battery', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            ship_marine = tg.create_task(hankyung_category(category='ship-marine', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            steel_chemical = tg.create_task(hankyung_category(category='steel-chemical', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            robot_future = tg.create_task(hankyung_category(category='robot-future', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            manage_business = tg.create_task(hankyung_category(category='manage-business', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            build_machinery = tg.create_task(hankyung_category(category='build-machinery', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_wallstreet_now = tg.create_task(hankyung_category(category='news-wallstreet-now', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_market = tg.create_task(hankyung_category(category='news-market', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_stock_focus = tg.create_task(hankyung_category(category='news-stock-focus', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_global_stock = tg.create_task(hankyung_category(category='news-global-stock', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_portfolio = tg.create_task(hankyung_category(category='news-portfolio', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_global_etf = tg.create_task(hankyung_category(category='news-global-etf', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_global_macro = tg.create_task(hankyung_category(category='news-global-macro', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_global_issue = tg.create_task(hankyung_category(category='news-global-issue', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_global_report = tg.create_task(hankyung_category(category='news-global-report', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_fed = tg.create_task(hankyung_category(category='news-fed', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_wti = tg.create_task(hankyung_category(category='news-wti', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_commodities_indices = tg.create_task(hankyung_category(category='news-commodities-indices', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_forex = tg.create_task(hankyung_category(category='news-forex', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            news_bond = tg.create_task(hankyung_category(category='news-bond', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))

        semicon_electronics = semicon_electronics.result()
        auto_battery = auto_battery.result()
        ship_marine = ship_marine.result()
        steel_chemical = steel_chemical.result()
        robot_future = robot_future.result()
        manage_business = manage_business.result()
        build_machinery = build_machinery.result()
        news_wallstreet_now = news_wallstreet_now.result()
        news_market = news_market.result()
        news_stock_focus = news_stock_focus.result()
        news_global_stock = news_global_stock.result()
        news_portfolio = news_portfolio.result()
        news_global_etf = news_global_etf.result()
        news_global_macro = news_global_macro.result()
        news_global_issue = news_global_issue.result()
        news_global_report = news_global_report.result()
        news_fed = news_fed.result()
        news_wti = news_wti.result()
        news_commodities_indices = news_commodities_indices.result()
        news_forex = news_forex.result()
        news_bond = news_bond.result()
        hankyung_result = semicon_electronics + auto_battery + ship_marine + steel_chemical + robot_future + manage_business + build_machinery + news_wallstreet_now + news_market + news_stock_focus + news_global_stock + news_portfolio + news_global_etf + news_global_macro + news_global_issue + news_global_report + news_fed + news_wti + news_commodities_indices + news_forex + news_bond
        print('hankyung 완료!')
        return hankyung_result

    @staticmethod
    async def maekyung(
                        end_datetime: str, date_format: str,
                        change_format: str, headers: dict[str, str],
                        min_delay: int | float = 2, max_delay: int | float = 3
                        ) -> list[dict[str, str]]:
        """maekyung 사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            end_datetime: 크롤링 및 스크래핑할 마지막 시각
            date_format: 시각 포맷
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
        
        async def maekyung_category(
                                        end_datetime: str, date_format: str,
                                        change_format: str, headers: dict[str, str], category: str,
                                        min_delay: int | float = 2, max_delay: int | float = 3
                                    ) -> list[dict[str, str, None]]:
            """maekyung 사이트를 크롤링 및 스크래핑 하는 메소드

            Args:
                end_datetime: 크롤링 및 스크래핑할 마지막 시각
                date_format: 시각 포맷
                change_format: 바꾸는 포맷
                headers: 식별 정보
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
            end_date = datetime.strptime(end_datetime, date_format)
            nonstop = True
            note = "마켓"
            maekyung_results = []

            while nonstop:
                time.sleep(random.uniform(min_delay, max_delay))
                web_page = maekyung_url(category=category, page=page)
                sync_result = sync_request(url=web_page, headers=headers)
                
                # html 문서 불러오기에 실패했으면 다음 페이지로 넘기기
                if sync_result['html'] is None:
                    print()
                    print(f'{page}번 페이지의 HTML 문서 정보를 불러오는데 실패했습니다.')
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

                async_result = await CrawlingScraping.async_crawling(url_list=url_list, category=category_dict[category], note=note, website=website, change_format=change_format, headers=headers)
                
                # 요청이 실패했으면 제외
                result = []
                for idx, res in enumerate(async_result):
                    if res is None:
                        print()
                        print(f'요청 실패한 데이터 : URL={url_list[idx]}, category={category}, website={website}')
                    else:
                        result.append(res)
                
                # end_date 이후가 아니면은 제거
                cut_info = datetime_cut(news_list=result, end_date=end_date, change_format=change_format)
                result, nonstop = cut_info['result'], cut_info['nonstop']

                maekyung_results.extend(result)
                page += 1
                print(f'website: {website}, category: {category_dict[category]}, page: {page}')
            print(f'website: {website}, category: {category_dict[category]} 종료')
            return maekyung_results
        
        async with asyncio.TaskGroup() as tg:
            company = tg.create_task(maekyung_category(category='company', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))

        company = company.result()
        maekyung_result = company
        print('maekyung 완료!')

        return maekyung_result

    @staticmethod
    async def yna(
                    end_datetime: str, date_format: str,
                    change_format: str, headers: dict[str, str],
                    min_delay: int | float = 2, max_delay: int | float = 3
                ) -> list[dict[str, str]]:
        """yna 사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            end_datetime: 크롤링 및 스크래핑할 마지막 시각
            date_format: 시각 포맷
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

        async def yna_category(
                                        end_datetime: str, date_format: str,
                                        change_format: str, headers: dict[str, str], category: str,
                                        min_delay: int | float = 2, max_delay: int | float = 3
                                    ) -> list[dict[str, str, None]]:
            """yna 사이트를 크롤링 및 스크래핑 하는 메소드

            Args:
                end_datetime: 크롤링 및 스크래핑할 마지막 시각
                date_format: 시각 포맷
                change_format: 바꾸는 포맷
                headers: 식별 정보
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
            end_date = datetime.strptime(end_datetime, date_format)
            nonstop = True
            if category in marketplus_set:
                note = '마켓+'
            yna_results = []

            async with async_playwright() as p:
                # 브라우저 열기(Chromium, Firefox, WebKit 중 하나 선택 가능)
                browser = await p.chromium.launch(headless=True)
                webpage = await browser.new_page()

                while nonstop:
                    # 웹사이트로 이동
                    time.sleep(random.uniform(min_delay, max_delay))
                    web_page = yna_url(category=category, page=page, marketplus_set=marketplus_set)
                    await webpage.goto(web_page)

                    news_container = webpage.locator('//*[@id="container"]/div/div/div[2]/section/div[1]/ul')
                    news_list = news_container.locator('//*[@class="news-con"]')
                    news_cnt = await news_list.count()

                    # 뉴스기사가 없으면 종료
                    if news_cnt == 0:
                        nonstop = False
                        break

                    news_list = await news_list.all()
                    url_list = [await news.locator("a").get_attribute("href") for news in news_list]

                    async_result = await CrawlingScraping.async_crawling(url_list=url_list, category=category_dict[category], note=note, website=website, change_format=change_format, headers=headers)
                
                    # 요청이 실패했으면 제외
                    result = []
                    for idx, res in enumerate(async_result):
                        if res is None:
                            print()
                            print(f'요청 실패한 데이터 : URL={url_list[idx]}, category={category}, website={website}')
                        else:
                            result.append(res)
                    
                    # end_date 이후가 아니면은 제거
                    cut_info = datetime_cut(news_list=result, end_date=end_date, change_format=change_format)
                    result, nonstop = cut_info['result'], cut_info['nonstop']

                    yna_results.extend(result)
                    page += 1
                    print(f'website: {website}, category: {category_dict[category]}, page: {page}')
                # 작업 후 브라우저 닫기
                await browser.close()
            print(f'website: {website}, category: {category_dict[category]} 종료')
            return yna_results
        
        async with asyncio.TaskGroup() as tg:
            global_stock = tg.create_task(yna_category(category='global-stock', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))
            global_market = tg.create_task(yna_category(category='global-market', end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=min_delay, max_delay=max_delay))

        global_stock = global_stock.result()
        global_market = global_market.result()
        yna_result = global_stock + global_market
        print('yna 완료!')
        return yna_result
    
    @staticmethod
    def web_crawling(
                    website: str, end_datetime: str,
                    date_format: str, change_format: str
                    ) -> list[dict[str, str, None]]:
        """해당 웹사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            website: 웹사이트 이름
            end_datetime: 크롤링 및 스크래핑할 마지막 시각
            date_format: 시각 포맷
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
                return asyncio.run(CrawlingScraping.hankyung(end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=3.5, max_delay=10))
            case 'maekyung':
                return asyncio.run(CrawlingScraping.maekyung(end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=3.5, max_delay=10))
            case 'yna':
                return asyncio.run(CrawlingScraping.yna(end_datetime=end_datetime, date_format=date_format, change_format=change_format, headers=headers, min_delay=3.5, max_delay=10))

    def run(self,
            end_datetime: str, date_format: str,
            change_format: str = '%Y-%m-%d %H:%M'
            ) -> dict[str, list[dict[str, str, None]]]:
        """멀티 프로세싱으로 웹사이트를 크롤링 및 스크래핑 하는 메소드

        Args:
            end_datetime: 크롤링 및 스크래핑할 마지막 시각
            date_format: 시각 포맷
            change_format: 바꾸는 포맷

        Returns:
            {
                "웹사이트1":    [
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
                "웹사이트2":    ...
                                    .
                                    .
                                    .
            }
        """

        website_list = list(self._crawling_scraping.keys())
        fixed_params_crawling = partial(CrawlingScraping.web_crawling, end_datetime=end_datetime, date_format=date_format, change_format=change_format)
        n = len(self._crawling_scraping)

        with futures.ProcessPoolExecutor(max_workers=n) as executor:
            for website, news_list in zip(website_list, executor.map(fixed_params_crawling, website_list)):
                self._crawling_scraping[website]._results = news_list
        
        return self._crawling_scraping

    def to_json(self) -> None:
        """크롤링 및 스크래핑한 데이터들을 json 파일로 저장하는 메소드"""

        for website in self._crawling_scraping.keys():
            self._crawling_scraping[website].to_json()