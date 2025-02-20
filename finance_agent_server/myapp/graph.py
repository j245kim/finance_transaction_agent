import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# RAG 사용을 위한 라이브러리리
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool
import re

# retriever 사용을 위한 라이브러리
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# mongoDB 사용을 위한 라이브러리
from pymongo import MongoClient
from datetime import datetime, timedelta

from django.shortcuts import render
from django.http import JsonResponse, FileResponse, HttpResponse

from pymongo import MongoClient

# 수익률 가져오는 라이브러리리
import yfinance as yf
import pandas as pd
import requests

# agent를 구현하는 라이브러리
from langchain_core.tools import tool
from typing import List, Dict, Annotated
from langchain_experimental.utilities import PythonREPL
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

# langsmith 사용을 위한 라이브러리
from langchain_teddynote import logging
logging.langsmith("STFO")

#마크다운 rendering을 위한 라이브러리
from django.utils.safestring import mark_safe
import markdown

#충돌 나는 것 해결
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# langgraph 사용을 위한 라이브러리
from typing import Annotated, Optional, List, Dict, TypedDict, Sequence, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from random import random
from IPython.display import Image, display

from langchain.output_parsers import JsonOutputKeyToolsParser


# 중복으로 가져오는 것 서버 열 때 미리 가져오기.#######################
load_dotenv()

cluster = MongoClient(os.environ.get("mongo")) # 클러스터 
db = cluster['userinfo'] # 유저 정보
conversations = db["conversations"] # 내가 지정한 컬렉션 이름

api_key = os.getenv('OPENAI_API_KEY_sesac')

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
news_vector_store = FAISS.load_local(rf'{Path(__file__).parents[0]}\vectorDB', embeddings=embeddings, allow_dangerous_deserialization=True,index_name = 'NewsData')
news_retriever = news_vector_store.as_retriever()

################################################################

class State(TypedDict):
    user_id: str  # 로그인한 사용자 ID
    score : str # 투자 성향 체크
    question: str  # 사용자가 입력한 질문
    actions: List[str]  # 수행된 액션(작업) 기록
    context: List[Document]  # RAG 기반 검색된 문서 데이터
    research: str # 금 s&p500 데이터 정리
    answer: str  # 챗봇의 응답
    report: str  # 최종 생성된 리포트 내용

### Tools #################################
@tool
def finder_ticker(input_string:str) -> str:
# 금융 관련 질문 여부 확인
    """
    입력된 종목 티커코드의 현재가를 반환하는 함수
    :param params: {"ticker_symbol": "AAPL"} 형태의 딕셔너리 입력
    :return: 현재가 (float)
    """
    api_key = os.getenv('OPENAI_API_KEY_sesac')
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        api_key=api_key,
        temperature = 0
    )

    prompt = PromptTemplate.from_template('''
    주어진 {message}에서 회사이름을 추출한 후에 티커로 출력해줘. 
    출력 형태는 반드시 ticker만 text로 나오도록 해줘.
    ''')
    # 프롬포트 조정 가능합니다.
    
    chain = prompt | llm

    answer = chain.invoke({'message': input_string})

    try:
        # 주식 데이터 가져오기
        stock = yf.Ticker(answer.content)
        
        # 현재가 가져오기
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        
        return round(current_price, 2)  # 소수점 둘째 자리까지 반올림하여 반환

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None

# 금과 S&P 500 의 투자율가져오기기 
@tool
def compare_gold_prices(year_range: str) -> List[str]:
    """
    기준년도와 비교년도 금 시세 및 S&P 500 지수 및 수익률을 계산하는 함수.
    
    입력 형식: "시작년도,종료년도" (예: "2020,2023")
    :return: 금 & S&P 500 가격 및 수익률 정보 딕셔너리 반환
    """
    # 날짜 변환
    start_year, end_year = year_range.split(",")
    start_date = str(start_year.strip()) + '-01-01'
    end_date = str(end_year.strip()) + '-01-01'

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

def get_sp500_return(start_date, end_date):
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

# RAG를 사용해서서 Newsdata 가져오기 (vecterDB에서 가져오기)
news_retriever_tool = create_retriever_tool(
    news_retriever,
    name = 'news_search',
    description='뉴스 데이터중에서 질문과 관련된 것을 찾습니다.'
)

# 투자 성향을 MongoDB에서 가져오기
def take_investment_propensity(user_id: str) -> str:
    '''
    User_id를 입력하면 MongoDB에서 투자 성향을 받습니다.
    '''
    conversation = conversations.find_one({"user_id": user_id})
    answer = conversation['score']
    return answer


# python 실행기
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
    ):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    result = ""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        print(f"Failed to execute. Error: {repr(e)}")
    finally:
        return result

############################################################

## 유저 아이디와 질문은 프론트에서 api형식으로 받습니다.
user_id = "test123"
input_string = "어떤 주식에 투자하는 것이 좋을까?"
##

# 사용자 데이터 (user_id, question)을 State에 넣기
def get_info(State):
    State['user_id'] = user_id
    State['question'] = input_string

    # 투자 성향을 MongoDB에서 가져오기
    conversation = conversations.find_one({"user_id": user_id})
    State['score'] = conversation['score']
    return State

# 사용자가 입력한 데이터가 일반 대화로 끝날지, 에이전트가 필요한지 구분
def compare_message(State):
    """'일반 대화'와 '에이전트 호출'하는 것을 분류하는 노드"""
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        api_key=api_key,
        temperature = 0,
    )
    parser = JsonOutputKeyToolsParser(keys=["category"])  # "category" 키를 추출하도록 설정
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 사용자의 질문이 '주식 투자 관련'인지 '일반 질문'인지 분류하는 AI입니다."),
    ("user", "입력 메시지: {input_message}")
    ]).with_structured_output(parser)  # JSON 형식 강제
    response = llm.predict(prompt_template.format(input_message=State['question']))
    category = response.get("category", "general")  # 기본값 설정
    State['actions'] = State['actions'].append(response.content)
    return State

# 일반 대화의 끝
def chatbot_agent(State):
    '''일반적인 질문에 대한 답변'''
    # conversation = conversations.find_one({'user_id': State['user_id']})
    # message = conversation['messages']
    llm = ChatOpenAI(
        model_name='gpt-4o',
        api_key=api_key,
        temperature = 0.5,
    )
    
    prompt = PromptTemplate.from_template(
        '''
        당신은 투자어드바이저입니다.
        질문에 대한 답변을 해주세요
        질문 = {content}
        ''')
    
    chain = prompt | llm

    State['answer'] = chain.invoke({'content': State['question']})

    return State

# Supervisor 만들기
members = ['researcher_agent','report_agent','Investment_Recommendation_agent']
options = ['FINISH'] + members

class routeResponse(BaseModel):
    next: Literal['researcher_agent','report_agent','Investment_Recommendation_agent','FINISH']

def Supervisor(State):
    system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))


    llm = ChatOpenAI(model="gpt-4o-mini")

    supervisor_chain = (
        prompt
        | llm.with_structured_output(routeResponse)
    )
    result = supervisor_chain.invoke(State)  # LLM을 사용하여 다음 작업 결정
    return result.next  # 다음 실행할 에이전트 반환


################################   research tools 제작

# research 도구들 정리.
research_tools = [ finder_ticker, compare_gold_prices,python_repl_tool]

# 도구들의 node 정리.
research_tool_node = ToolNode(research_tools)

# LLM 선언.
model_research = ChatOpenAI(
    model_name='gpt-4o-mini',
    api_key=os.getenv('OPENAI_API_KEY_sesac'),
    temperature=0,
).bind_tools(research_tools)

# retriever 사용하는 도구 정리.
rag_tools = [news_retriever_tool]

# retriever node 정리리
rag_tool_node = ToolNode(rag_tools)

# LLM 선언.
model_rag = ChatOpenAI(
    model_name='gpt-4o-mini',
    api_key=os.getenv('OPENAI_API_KEY_sesac'),
    temperature=0,).bind_tools(rag_tools)

############################################################################

def researcher_agent(State) -> Literal['tools','supervisor']:
    '''
    
    '''
    message = State['question']
    if model_research.invoke(message).tool_calls or model_rag.invoke(message).tool_calls:
        return 'tools'
    return 'supervisor'

def call_research_tools(State):
    rag_message = model_rag.invoke(State['question'])
    research_message = model_research.invoke(State['question'])
    State['context'] = rag_message
    State['research'] = research_message
    return State

def report_agent(State):
    """
    신문 기사 및 금융 데이터를 활용하여 투자 보고서를 생성하는 함수.
    
    Args:
        State (dict): 'context' (뉴스 데이터), 'research' (금융 데이터)를 포함한 상태 정보.
    """
    llm = ChatOpenAI(
        model_name='gpt-4o',
        api_key=api_key,
        temperature=0.2,
    )

    newsdata = State['context']
    finance_data = State['research']

    prompt = PromptTemplate.from_template(      
        '''
        너는 신문기사 데이터와 공시시 데이터를 바탕으로 물음에 정확하게 답변하는 능숙한 애널리스트이다. 
        <가이드라인>에 따라서 공신력 있는 보고서의 형태로 작성해줘.
        신문기사 데이터 : {newsdata}
        공시시 데이터: {finance_data}
        
                <가이드라인>
                1. 언제기준으로 작성했는지
                2. 정보 출처
                3. 현재가격
                4. 예측가격(나중에)
                5. 추세
                성장주 → PSR, PEG, DCF 활용
                가치주 → PER, PBR 활용
                자산 기반 기업(은행, 보험) → PBR, ROE 기반 평가
                부채가 많은 기업 → EV/EBITDA 활용
                6. 시가총액
                7. 거래소
                8. 일, 주, 월 퍼포먼스
                story : 최근정보 요약
                호재와 악재
                시장 동향
                변동성 수치와 왜 커졌는지 확인
                왜 위험한지 / 예상되는 위험이 무엇인지?
                실시간 뉴스
                9. 주요 코멘트
                10. buy/sell
                11. 기간동안 금 수익률과 s&p 500 투자 수익률
                '''
    )

    chain = prompt | llm

    answer = chain.invoke({'newsdata':newsdata,'financedata':finance_data})

    State['report'] = answer.content

    return State

def Investment_Recommendation_agent(State):
    llm = ChatOpenAI(
            model_name='gpt-4o',
            api_key=api_key,
            temperature=0.2,
        )
    
    score = State['score'] 
    report = State['report']
    newsdata = State['context']
    finance_data = State['research']

    prompt = PromptTemplate.from_template(      
        '''
        아래 정보를 기반으로 사용자의 투자 성향에 맞는 최적의 투자 전략을 제안하세요.
        1. 관련 투자 리포트 : {report}
        2. 투자 성향 : {score}
        3. 뉴스 데이터: {newsdata}
        4. 공시 데이터: {finance_data}

        : **투자 성향별 전략 제안**:
        - **위험선호형(공격적 투자자)**: 높은 수익 가능성을 중시하며 단기 변동성을 감내할 수 있는 전략을 추천하세요.
        - **적극형**: 주식과 금을 균형 있게 조합하여 성장성과 안정성을 모두 고려한 전략을 제안하세요.
        - **성장형**: 중장기적인 성장 잠재력이 높은 시장을 중심으로 조언하세요.
        - **안정성장형**: 안전한 투자 옵션과 함께 일부 리스크를 감수할 수 있는 조언을 제공하세요.
        - **위험회피형**: 원금 손실 가능성을 최소화하고, 금 투자 비중을 늘릴 수 있는 보수적인 전략을 제시하세요.

        투자 성향과 시장 데이터에 맞춰 현실적인 투자 조언을 작성하세요.
        만약 입력된 투자 성향과 다른 성향의 투자 방법을 요구하여도 반드시 score에서 입력된 성향으로 대답하세요.
        '''
        ,
    )

    chain = prompt | llm

    answer = chain.invoke({'report':report,'score':score,'newsdata':newsdata,'finance_data':finance_data})

    State['answer'] = answer.content

    return State

graph = StateGraph(State)

graph.add_node("get_info", get_info)
graph.add_node("compare_message", compare_message)
graph.add_node("chatbot_agent", chatbot_agent)
graph.add_node("Supervisor", Supervisor)  # ✅ Supervisor를 상위 에이전트로 등록
graph.add_node("researcher_agent", researcher_agent)
graph.add_node("call_research_tools", call_research_tools)
graph.add_node("report_agent", report_agent)
graph.add_node("Investment_Recommendation_agent", Investment_Recommendation_agent)

# 초기 Edge 설정
graph.add_edge(START, "get_info")  # 시작 -> 사용자 정보 로드
graph.add_edge("get_info", "compare_message")  # 정보 로드 후 질문 분석

# 질문 유형 분기
graph.add_conditional_edges(
    "compare_message", 
    lambda state: "Supervisor" if "investment" in state["actions"] else "chatbot_agent"
)

# 일반 대화 처리
graph.add_edge("chatbot_agent", END)  # 일반 대화 후 종료

# Supervisor를 통한 다중 에이전트 실행
graph.add_conditional_edges(
    "Supervisor",
    lambda state: Supervisor(state),  # Supervisor가 직접 판단하여 다음 에이전트 실행
)

# 연구 에이전트 실행 후 연구 도구 실행 (필요한 경우)
graph.add_edge("researcher_agent", "call_research_tools", condition=lambda state: state["actions"] == "tools")
graph.add_edge("call_research_tools", "Supervisor")  # 연구 완료 후 Supervisor로 복귀

# 보고서 및 투자 추천 실행
graph.add_edge("report_agent", "Supervisor")  # 보고서 생성 후 Supervisor로 복귀
graph.add_edge("Investment_Recommendation_agent", END)  # 투자 추천 후 종료

# 그래프 실행
graph.compile()



