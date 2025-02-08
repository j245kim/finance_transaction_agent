# -*- coding: utf-8 -*-

# 파이썬 표준 라이브러리
import random
import time
import asyncio

# 파이썬 서드파티 라이브러리
import httpx


def sync_request(
                url: str, headers: dict[str, str], follow_redirects: bool = True,
                timeout: int | float = 90, encoding: str = 'utf-8', max_retry: int = 10,
                min_delay: int | float = 2, max_delay: int | float = 3
                 ) -> dict[str, int, str, httpx.Response, None]:
    """동기로 HTML 문서 정보를 불러오는 함수

    Args:
        url: URL
        headers: 식별 정보
        follow_redirects: 리다이렉트 허용 여부
        timeout: 응답 대기 허용 시간
        encoding: 인코딩 방법
        max_retry: HTML 문서 정보 불러오기에 실패했을 때 재시도할 최대 횟수
        min_delay: 재시도 할 때 딜레이의 최소 시간
        max_delay: 재시도 할 때 딜레이의 최대 시간
    
    Return:
        {
            "html": HTML 문서 정보, str | None
            "response_status_code": 응답 코드, int
            "response_reason": 응답 결과 이유, str
            "response_history": 수행된 redirect 응답 목록, list[Response]
        }
    """

    result = {"html": None, "response_status_code": None, "response_reason": None, "response_history": None,
              "error_type": None}

    with httpx.Client(headers=headers, follow_redirects=follow_redirects, timeout=timeout, default_encoding=encoding, limits=httpx.Limits(max_keepalive_connections=150, max_connections=150)) as client:
        for _ in range(max_retry):
            try:
                # 동기 client로 HTML GET
                response = client.get(url)
                # 응답 기록 추가
                result['response_status_code'] = response.status_code
                result['response_reason'] = response.reason_phrase
                result['response_history'] = response.history
                # HTML 문서 정보를 불러오는 것에 성공하면 for문 중단
                if response.status_code == httpx.codes.ok:
                    result['html'] = response.text
                    break

                # 동기 제어 유지(멀티 프로세싱이라는 전제)
                time.sleep(random.uniform(min_delay, max_delay))
            except Exception as e:
                print()
                print(f'{url}에서 {type(e).__name__}가 발생했습니다.')
                # 응답 기록 추가
                result['response_status_code'] = None
                result['response_reason'] = None
                result['response_history'] = None
                result['error_type'] = type(e).__name__
    
    return result


async def async_request(
                        url: str, headers: dict[str, str], follow_redirects: bool = True,
                        timeout: int | float = 90, encoding: str = 'utf-8', max_retry: int = 10,
                        min_delay: int | float = 2, max_delay: int | float = 3
                        ) -> dict[str, int, str, httpx.Response, None]:
    """비동기로 HTML 문서 정보를 불러오는 함수

    Args:
        url: URL
        headers: 식별 정보
        follow_redirects: 리다이렉트 허용 여부
        timeout: 응답 대기 허용 시간
        encoding: 인코딩 방법
        max_retry: HTML 문서 정보 불러오기에 실패했을 때 재시도할 최대 횟수
        min_delay: 재시도 할 때 딜레이의 최소 시간
        max_delay: 재시도 할 때 딜레이의 최대 시간
    
    Return:
        {
            "html": HTML 문서 정보, str | None
            "response_status_code": 응답 코드, int
            "response_reason": 응답 결과 이유, str
            "response_history": 수행된 redirect 응답 목록, list[Response]
        }
    """

    result = {"html": None, "response_status_code": None, "response_reason": None, "response_history": None,
              'error_type': None}

    async with httpx.AsyncClient(headers=headers, follow_redirects=follow_redirects, timeout=timeout, default_encoding=encoding, limits=httpx.Limits(max_keepalive_connections=200, max_connections=200)) as client:
        for _ in range(max_retry):
            try:
                # 비동기 client로 HTML GET
                response = await client.get(url)
                # 응답 기록 추가
                result['response_status_code'] = response.status_code
                result['response_reason'] = response.reason_phrase
                result['response_history'] = response.history
                # HTML 문서 정보를 불러오는 것에 성공하면 for문 중단
                if response.status_code == httpx.codes.ok:
                    result['html'] = response.text
                    break

                # 비동기 코루틴 제어 양도
                await asyncio.sleep(random.uniform(min_delay, max_delay))
            except Exception as e:
                print()
                print(f'{url}에서 {type(e).__name__}가 발생했습니다.')
                # 응답 기록 추가
                result['response_status_code'] = None
                result['response_reason'] = None
                result['response_history'] = None
                result['error_type'] = type(e).__name__
    
    return result