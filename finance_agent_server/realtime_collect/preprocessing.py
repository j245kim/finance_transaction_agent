# -*- coding: utf-8 -*-

# 파이썬 표준 라이브러리
import re
from datetime import datetime
from copy import deepcopy

# 파이썬 서드파티 라이브러리
from bs4 import BeautifulSoup, Comment


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
                news_list: list[dict[str, str | None]],
                end_date: datetime, change_format: str
                ) -> dict[str, list[dict[str, str | None]] | bool]:
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


# 각 사이트별 본문 전처리 함수
def hankyung_atag_remove(x: re.Match) -> str:
    """한국경제 본문에서 a태그에서 텍스트만 남기고 나머지 태그 정보등을 제거하는 전처리 함수

    Args:
        x: 매칭된 문자열

    Returns:
        전처리된 문자열
    """

    x = x.group(1)
    x = re.sub(pattern=r' +', repl=' ', string=x)
    return x    


def hankyung_preprocess(data_list: list[dict[str, str]]) -> None:
    """한국경제 본문을 전처리 하는 함수

    Args:
        뉴스 데이터 리스트
    
    Returns:
        None
    """

    for data in data_list:
        soup = BeautifulSoup(data['news_content'], 'html.parser')
        content = []

        # ------------------------------------------------------------------------------------------------------------------------------------

        # BeautifulSouo로 일부 tag 제거
        # 동영상 제거
        for tag in soup.find_all("div", {"class": "article-video"}):
            tag.decompose()

        # 이미지 제거
        for tag in soup.find_all("figure"):
            tag.decompose()

        # 스크립트 제거
        for tag in soup.find_all(lambda tag: tag.name == 'script'):
            tag.decompose()
        
        # 처음 div class="article-body"를 제외한 나머지 div들을 제거
        if (tag := soup.find(lambda tag: tag.name == "div" and tag.has_attr("style"))):
            tag.unwrap()
        outer_div = soup.find("div")
        for inner_div in outer_div.find_all("div", recursive=False):
            inner_div.unwrap()
        
        # ------------------------------------------------------------------------------------------------------------------------------------
        # 만약 article tag가 있으면 article tag안으로 들어가기
        if (inner := soup.find('article')) is not None:
            inner = inner.contents
        else:
            inner = soup.div.contents

        # 기자 이름 및 이메일 제거
        pattern_name_email = r'\w+@'
        content = [str(s) for s in inner if re.search(pattern=rf'({pattern_name_email})', string=s.text, flags=re.IGNORECASE) is None]

        # 양 끝자락에 있는 화이트스페이스 제거
        content = ''.join(content).strip()

        # 내부 div들을 다시 제거
        content = re.sub(pattern=r'<div .*?(</div>\s*)+', repl='', string=content)

        # amp; 제거
        content = content.replace('amp;', '')

        # a태그 및 하이퍼링크 제거
        content = re.sub(pattern=r'\s*?<a href=.*?>\n(.+?)\n\s+</a>\s+', repl=hankyung_atag_remove, string=content)

        # em태그 제거
        content= re.sub(pattern=r'\s*?<em>\n(.+?)\n\s+</em>\s+', repl=hankyung_atag_remove, string=content)

        # 오른쪽 끝자락에 있는 r'\s*?(<br/>\s*?)+?$'패턴들을 제거
        content = re.sub(pattern=r'\s*?(<br/>\s*?)+?$', repl='', string=content)

        # 왼쪽 끝자락에 있는 ^\s*?(<br/>\s*?)+?패턴들을 제거
        content = re.sub(pattern=r'^\s*?(<br/>\s*?)+?', repl='', string=content)

        # 기자 이름 제거
        content = re.sub(pattern=r'[\s가-힣]*? 기자$', repl='', string=content)

        # 오른쪽 끝자락에 있는 r'\s*?(<br/>\s*?)+?$'패턴들을 제거
        content = re.sub(pattern=r'\s*?(<br/>\s*?)+?$', repl='', string=content)

        # 양 끝자락에 있는 화이트스페이스 제거
        content = content.strip()

        # 오른쪽 끝자락에 있는 <br/> <br class 등 제거
        content = re.sub(pattern=r'\s*<br/>\s*<br [^가-힣]*?>$', repl='', string=content)

        # 양 끝자락에 있는 화이트스페이스 제거
        content = content.strip()

        data['news_content'] = content
    return None


def maekyung_preprocess(data_list: list[dict[str, str]]) -> None:
    """매일경제 본문을 전처리 하는 함수

    Args:
        뉴스 데이터 리스트
    
    Returns:
        None
    """

    for data in data_list:
        soup = BeautifulSoup(data['news_content'], 'html.parser')

        # ------------------------------------------------------------------------------------------------------------------------------------

        # BeautifulSouo로 일부 tag 제거
        # 이미지 제거
        for tag in soup.find_all(lambda tag: tag.name == 'div' and tag.has_attr('style') and tag['style'] == 'width:100%; text-align: center;'):
            tag.decompose()

        # 스크립트 제거
        for tag in soup.find_all(lambda tag: tag.name == 'script'):
            tag.decompose()

        # 주석 제거
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # MK시그널 링크 제거
        for tag in soup.find_all('a', {"class": "ix-editor-text-link"}):
            tag.decompose()

        # ------------------------------------------------------------------------------------------------------------------------------------

        # replace 및 정규표현식으로 제거
        # 본문 태그 제거 및 양 끝에 있는 \n같은 화이트스페이스 제거
        content = [str(s) for s in soup.div.contents]
        content = ''.join(content).strip()

        # [김진우 기자], [김진우 기자 / 권용희 기자] 같은 불필요한 정보 제거
        content = re.sub(pattern=r'\[.+?/?.+?기자\]', repl='', string=content)

        # MK시그널의 쓰잘데기 없는 광고 문구 제거
        content = re.sub(pattern=r'※ 매일경제 AI기자.*?앱을 다운로드하세요\!', repl='', string=content)
        content = re.sub(pattern=r'이 종목의 향후 매매신호를 받으려면 MK시그널.*?실시간으로 확인할 수 있습니다\.', repl='', string=content)
        content = re.sub(pattern=r'MK시그널은 AI가 국내·미국 주식.*?정보 서비스입니다\.', repl='', string=content)
        content = re.sub(pattern=r'국내 주식과 더불어 미국 주식.*?신호 받아보세요\!', repl='', string=content)
        content = re.sub(pattern=r'인공지능\(AI\) 기반 매매신호 제공 앱.*?투자를 시작하세요\!', repl='', string=content)
        content = re.sub(pattern=r'MK시그널 현재.*?진행중\!', repl='', string=content)

        # '나현준의 데이터로 세상 읽기'에서 불필요한 정보 제거
        content = re.sub(pattern=r'◆\s*매경 포커스\s*◆', repl='', string=content)

        # amp; 제거
        content = content.replace('amp;', '')

        # 과도한 <br/>\n들을 줄이기
        content = re.sub(pattern=r'(\s+<br/>){3,}', repl=r"\n <br/>\n<br/>", string=content)

        # 양 끝자락에 있는 화이트스페이스 제거
        content = content.strip()

        # 오른쪽 끝자락에 있는 r'\s*?(<br/>\s*?)+?$'패턴들을 제거
        content = re.sub(pattern=r'\s*?(<br/>\s*?)+?$', repl='', string=content)

        # 왼쪽 끝자락에 있는 ^\s*?(<br/>\s*?)+?패턴들을 제거
        content = re.sub(pattern=r'^\s*?(<br/>\s*?)+?', repl='', string=content)

        # 기자 이름 제거
        content = re.sub(r'[가-힣]+ 기자$', repl='', string=content)

        # 오른쪽 끝자락에 있는 r'\s*?(<br/>\s*?)+?$'패턴들을 제거
        content = re.sub(pattern=r'\s*?(<br/>\s*?)+?$', repl='', string=content)

        # 왼쪽 끝자락에 있는 ^\s*?(<br/>\s*?)+?패턴들을 제거
        content = re.sub(pattern=r'^\s*?(<br/>\s*?)+?', repl='', string=content)

        # 양 끝자락에 있는 화이트스페이스 제거
        content = content.strip()

        data['news_content'] = content
    return None


def yna_preprocess(data_list: list[dict[str, str]]) -> None:
    """연합뉴스 본문을 전처리 하는 함수

    Args:
        뉴스 데이터 리스트
    
    Returns:
        None
    """

    for data in data_list:
        soup = BeautifulSoup(data['news_content'], 'html.parser')
        content = []

        # ------------------------------------------------------------------------------------------------------------------------------------

        # BeautifulSouo로 일부 tag 제거
        # 기자 이름 제거
        for tag in soup.find_all(lambda tag: tag.has_attr('id') and re.search(pattern=r'newsWriter', string=tag['id']) is not None):
            tag.decompose()

        # 이미지 제거
        for tag in soup.find_all("div", {"class": "comp-box photo-group"}):
            tag.decompose()

        # 동영상 제거
        for tag in soup.find_all("div", {"class": "comp-box youtube-group"}):
            tag.decompose()

        # 중간 광고 제거
        for tag in soup.find_all("aside"):
            tag.decompose()

        # 관련 기사 제거
        for tag in soup.find_all("div", {"class": "related-zone rel"}):
            tag.decompose()

        # 제보 및 저작권 제거
        for tag in soup.find_all("p", {"class": "txt-copyright adrs"}):
            tag.decompose()

        # ------------------------------------------------------------------------------------------------------------------------------------

        # 기자 이름 및 이메일 제거
        # 하이퍼링크 제거
        pattern_name_email = r'\w+@'

        for s in soup.article.contents:
            if re.search(pattern=rf'({pattern_name_email})', string=s.text, flags=re.IGNORECASE) is not None:
                continue

            content.append(str(s))

        # 양 끝자락에 있는 화이트스페이스 제거
        content = ''.join(content).strip()

        # amp; 제거
        content = content.replace('amp;', '')

        # a태그 및 하이퍼링크 제거
        content = re.sub(pattern=r'\n\s*<span class=.*?>\s*<a .*?>\s*(.*?)\s*</a>\s*</span>\s*', repl=r' \g<1>', string=content)

        # 끝자락에 있는 <br/> 제거
        content = re.sub(pattern=r'<p>\s*?<br/>\s*?</p>$', repl='', string=content)

        # 양 끝자락에 있는 화이트스페이스 제거
        content = content.strip()

        data['news_content'] = content
    return None