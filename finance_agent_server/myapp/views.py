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
from langchain_core.prompts import ChatPromptTemplate

# mongoDB 사용을 위한 라이브러리
from pymongo import MongoClient
from datetime import datetime, timedelta

from django.shortcuts import render
from django.http import JsonResponse, FileResponse, HttpResponse

# 수익률 가져오는 라이브러리리
import yfinance as yf
import pandas as pd
import requests

# agent를 구현하는 라이브러리
from langchain.tools import tool
from typing import List, Dict, Annotated
from langchain_experimental.utilities import PythonREPL
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor



# langsmith 사용을 위한 라이브러리
from langchain_teddynote import logging
load_dotenv()
logging.langsmith("STFO")

#마크다운 rendering을 위한 라이브러리
from django.utils.safestring import mark_safe
import markdown


#충돌 나는 것 해결
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from django.views.decorators.csrf import csrf_exempt

# 중복되는 항목
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY_sesac')
cluster = MongoClient(os.environ.get("mongo")) # 클러스터 

# HTML창 띄우기
from django.shortcuts import render

@csrf_exempt
def chatgpt(req, user_id ,input_string):
    db = cluster['userinfo'] # 유저 정보
    conversations = db["conversations"] # 내가 지정한 컬렉션 이름

    llm = ChatOpenAI(
        model_name='gpt-4o',
        api_key=api_key
    )

    if conversations.find_one({"user_id": user_id}) == None:
        conversations.insert_one({"user_id": user_id,'score':[],'stack': [], 'ban':[] , "messages": []})  # 새로운 유저 데이터 추가

    user = conversations.find_one({"user_id": user_id})

    # 정해진 시간 만큼지나면 삭제하는 코드
    # pull을 통해 stack에서 삭제
    # 현재는 1시간이 지나면 삭제됨
    
    time_threshold = datetime.now() - timedelta(hours=1)
    conversations.update_one(
        {"user_id": user_id},
        {"$pull": {"stack": {"timestamp": {"$lt": time_threshold}}}}
    )


    # 금융 관련 질문이 아닌 횟수가 5번 이상일 시에 밴
    if user and len(user['stack']) >= 5:
        conversations.update_one(
            {"user_id": user_id},
            {'$push': {'ban': datetime.now()}}
        )
        return JsonResponse({'content':'금융 관련된 질문을 하지 않아서 밴되었습니다. 1시간 후에 다시 시도해주세요.'})


    # 금융 관련 질문인지 확인하고 아닐 시에 경고 횟수와 시간을 함께 MongoDB에 저장
    # push를 통해 stack에 저장
    if classification_finance(input_string) == 'False':
        conversations.update_one(
            {"user_id": user_id},
            {"$push": {"stack": {'value':1 ,'timestamp': datetime.now()}}}
        )
        return JsonResponse({'content':'금융 관련 질문이 아닙니다. 금융 관련 질문을 해주세요.'})
    
    conversation = conversations.find_one({"user_id": user_id})

    messages = conversation["messages"]  # 기존 메시지 리스트 가져오기

    messages.append({"role": "user", "content": input_string})  # 새 메시지 추가

    if classify_intent(input_string) == '일반 금융 정보':
        # GPT에서 질문을 받아서 대답해주게 만들기.
        prompt = PromptTemplate.from_template(
            '''
            당신은 투자어드바이저입니다. 
            질문에 대한 답변을 해주세요.
            질문 = {content}
            ''')
    
        chain = prompt | llm

        answer = chain.invoke(input={"content": input_string})

        messages.append({"role": "assistant", "content": answer.content})

        conversations.update_one(
            {"user_id": user_id},
            {"$set": {"messages": messages}}
        )
        
        return JsonResponse({'content':answer.content})

    if classify_intent(input_string) == '보고서 필요':
        # 리포트 제작을 위해서 가져옵시다.
        answer = make_and_save_report(user_id,input_string)

        messages.append({"role": "assistant", "content": '보고서를 작성하였습니다.'})

        conversations.update_one(
            {"user_id": user_id},
            {"$set": {"messages": messages}}
        )

        return JsonResponse({'content':answer})

    if classify_intent(input_string) == '주식 매매 필요':
        db = cluster['userinfo']
        user_info = db['info']
        sellect_info = user_info.find_one({"username": user_id})

        if not isinstance(sellect_info, dict):
            return JsonResponse({'content':'a: 거래소 ID, 계좌 번호, app_key, app_secret_key를 왼쪽 하단에 개인 정보를 통하여 등록하세요!'})
        
        if not (
            sellect_info.get('거래소ID') and
            sellect_info.get('account_number') and
            sellect_info.get('mock_app_key') and
            sellect_info.get('mock_app_secret')
            ):
            return JsonResponse({'content': 'b: 거래소 ID, 계좌 번호, app_key, app_secret_key를 왼쪽 하단에 개인 정보를 통하여 등록하세요!'})

        return JsonResponse({"content": f"<a href='/trading/' target='_blank'>이 곳을 클릭해서 거래를 완료하세요!</a>"})

    else:
        # 투자 성향 체크
        if conversation['score'] == []:    
            return JsonResponse({'content':'투자 성향 체크가 필요합니다! 왼쪽 사이드바에서 투자 성향 체크를 클릭해주세요!. 투자 성향 체크가 끝났다면, 다시 질문해주세요!'})
        else :
            # 투자 성향을 이용하여 투자 추천.
            answer = compare_invest(user_id, input_string)
        
            return JsonResponse({'content':answer})

@csrf_exempt
def classify_intent(user_input):
    """
    사용자의 입력을 분석하여 해당하는 기능을 반환하는 함수.
    """
    llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    api_key=api_key,
    temperature = 0.5
    )

    prompt = PromptTemplate.from_template(
        """
    1. **"보고서 필요"**  
    - 사용자가 **주식 리서치 보고서**를 요청하는 경우  
    - 예시: "애플의 최근 실적 분석 보고서 작성해줘.", "삼성전자 주식 보고서 줘."  
    - 키워드 : "리포트", "보고서", "분석 자료", "pdf 생성", "보고서 작성"

    2. **"투자 권유 필요"**  
    - 사용자가 **투자 결정을 도와달라고 요청**하는 경우  
    - 예시: "애플 주식 지금 사야 할까?", "테슬라 투자 추천해줘.", "코인이랑 주식 중 뭐가 나을까?"  
    - 키워드 : "추천", "매수", "매도", "포트폴리오", "어떤 주식", "살까"

    3. **"일반 금융 정보"**  
    - 간단한 금융 정보나 주가 확인, 시장 동향 질문  
    - 예시: "삼성전자 주가 알려줘.", "애플 주식 전망이 어때?", "코스피 지수 어떻게 돼?"  

    4. **"주식 매매 필요"**
    - 사용자가 **주식의 매수, 매도, 잔고 확인**을 요청하는 경우
    - 예시: "애플 주식을 2주 사고 싶어", "인텔 주식 4주 팔고 싶어", "내가 가지고 있는 주식 잔고를 확인하고 싶어"
    - 키워드 : "매수", "매도", "잔고"
    
    반드시 아래 형식식으로 답변하세요.

    {{"classification": "투자 권유 필요"}}

    {{"classification": "보고서 필요"}}

    {{"classification": "일반 금융 정보"}}

    {{"classification": "주식 매매 필요"}}


    사용자 입력: {user_input}
    """
    )

    chain = prompt | llm

    # LLM 실행
    response = chain.invoke(user_input)

    # JSON 파싱 (에러 처리 포함)
    try:
        parsed_response = json.loads(response.content)
        classification = parsed_response.get("classification", "일반 금융 정보")  # 기본값 설정
    except json.JSONDecodeError:
        classification = "일반 금융 정보"  # JSON이 깨졌을 경우 기본값 설정

    return classification

@csrf_exempt
def classification_finance(input_string):
# 금융 관련 질문 여부 확인
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY_sesac')
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        api_key=api_key,
        temperature = 0.5
    )

    prompt = PromptTemplate.from_template('''
    message : '{message}'

    금융으로 이해할 수 있는 어휘 또는 표현은 금융 관련 질문이라고 간주하세요.
    금융(예: 주식, 투자, 은행, 대출, 경제 등)과 직접 관련이 있는 경우에만 "True"를 반환하고, 그렇지 않다면 "False"를 반환하세요.
    답변은 "True" 또는 "False"만 출력하세요.
    ''')
    # 프롬포트 조정 가능합니다.
    
    chain = prompt | llm

    answer = chain.invoke({'message': input_string})

    return answer.content

@csrf_exempt
#백터 DB에서 retriever로 찾아주는 함수 (사용하지 않음)
def invest_search_rag(user_id,input_string):
    # RAG 제작

    load_dotenv()

    cluster = MongoClient(os.environ.get("mongo")) # 클러스터 
    db = cluster['userinfo'] # 유저 정보
    conversations = db["conversations"] # 내가 지정한 컬렉션 이름

    conversation = conversations.find_one({"user_id": user_id})

    messages = conversation["messages"]  

    api_key = os.getenv('OPENAI_API_KEY_sesac')
    
    llm = ChatOpenAI(
        model_name='gpt-4o',
        api_key=api_key,
        max_tokens=500,
    )

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # faiss에 저장
    if not os.path.isfile(rf'{Path(__file__).parent}/vectorDB/NewsData.faiss') :
        
        # mongoDB에서 데이터를 가져오기
        cluster = MongoClient(os.environ.get("mongo")) # 클러스터 
        db = cluster['Document'] # 유저 정보
        newsdata = db["newsdata"] # 내가 지정한 컬렉션 이름

        newsdatas = newsdata.find().to_list()

        docs = [Document(page_content = news_info['news_content'],metadata = {
                                                                                'title': news_info['news_title'],
                                                                                'first_time': news_info['news_first_upload_time'],
                                                                                'last_time':news_info['news_last_upload_time'],
                                                                                'subcategory':news_info['news_category'],
                                                                                'major category':news_info['note'],
                                                                                'website':news_info['news_website'],
                                                                                'url':news_info['news_url'],
                                                                                'author':news_info['news_author'],
                                                                                })
            for news_info in newsdatas]


        # 위와 같은 이유로 전처리하는 것을 찾아보겠음.
        
        # 찾아보니까 임베딩 모델이 openaiembeddings를 사용하고 있어서 토큰화가 필요없다. 자동으로 하긴하는데, 전처리를 해주는 것은 좋아보인다.
        # 전처리를 하고 임베딩 모델에 넣어서 faiss에 저장해보자.
        
        # 청크로 나누기 
        splitter = CharacterTextSplitter(chunk_size = 50, chunk_overlap = 5)
        split_texts = splitter.split_documents(docs)

        # faiss로 벡터db 저장장
        news_vector_store = FAISS.from_documents(split_texts, embeddings)
    
        news_vector_store.save_local(rf'{Path(__file__).parents[0]}/vectorDB',index_name = 'NewsData')
    
    # 백터db 로드
    news_vector_store = FAISS.load_local(rf'{Path(__file__).parents[0]}/vectorDB', embeddings=embeddings, allow_dangerous_deserialization=True,index_name = 'NewsData')

    # retriever 생성
    # 유사도 검색을 코사인 유사도가 아닌 다른 방법으로 하고 싶다면 as.retriever의 인자를 바꾸면 된다.
    news_retriever = news_vector_store.as_retriever()

    news_retriever_tool = create_retriever_tool(
        news_retriever,
        name = 'news_search',
        description='뉴스 데이터중에서 질문과 관련된 것을 찾습니다.'
    )
    
    # agent > tool 준비
    tools = [news_retriever_tool, ]

    # 프롬포트 조정 가능합니다.
    prompt = ChatPromptTemplate.from_messages(

        [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `compare_gold_prices` tool to compare returns and provide investment solutions.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Agent 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

        # AgentExecutor 생성
    agent_executor = AgentExecutor(
        agent=agent,                  # 1) 사용할 에이전트
        tools=tools,                  # 2) 사용할 도구 목록
        verbose=True,                  # 3) 디버깅 메시지 출력 여부
        max_iterations=10,             # 4) 최대 실행 반복 횟수
        max_execution_time=100,         # 5) 최대 실행 시간 제한
        handle_parsing_errors=True,    # 6) 파싱 오류 처리 여부
    )
    # AgentExecutor 실행
    result = agent_executor.invoke({"input": input_string})

    messages.append({"role": "assistant", "content": result['output']})

    conversations.update_one(
        {"user_id": user_id},
        {"$set": {"messages": messages}}
    )
    print('retriever 사용함.')
    return result['output']

@csrf_exempt
#mongodb에 atype저장용
def measure_investment_propensity(request,user_id,score):
    load_dotenv()

    cluster = MongoClient(os.environ.get("mongo")) # 클러스터 
    db = cluster['userinfo'] # 유저 정보
    conversations = db["conversations"] # 내가 지정한 컬렉션 이름

    atype = ''
    if int(score) >= 26:
        atype = '위험 선호형'
    elif int(score) >= 21:
        atype = '적극 투자형'
    elif int(score) >= 16:
        atype = '성장 투자형'
    elif int(score) >= 11:
        atype = '안정 성장형'
    else :
        atype = '안정형'



    conversations.update_one(
        {'user_id':user_id},
        {'$set':{'score':atype}},
    )
    return JsonResponse({'content':'투자 성향 체크를 다시 하셨습니다!'})

@csrf_exempt
#투자 성향에 따라 투자를 권유하는 기능
def compare_invest(user_id, input_string):
    api_key = os.getenv('OPENAI_API_KEY_sesac')
    
    llm = ChatOpenAI(
        model_name='gpt-4o',
        api_key=api_key,
    )

    # mongoDB에서 데이터를 가져오기
    cluster = MongoClient(os.environ.get("mongo")) # 클러스터 
    db = cluster['userinfo'] # 유저 정보
    conversations = db["conversations"] # 내가 지정한 컬렉션 이름

    conversation = conversations.find_one({"user_id": user_id})

    messages = conversation["messages"]  # 기존 메시지 리스트 가져오기

    score = conversation['score']

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # 백터db 로드
    # NewsData.faiss
    news_vector_store = FAISS.load_local(rf'{Path(__file__).parents[0]}/vecterDB', embeddings=embeddings, allow_dangerous_deserialization=True,index_name = 'NewsData')
    # retriever 생성
    # 유사도 검색을 코사인 유사도가 아닌 다른 방법으로 하고 싶다면 as.retriever의 인자를 바꾸면 된다.
    news_retriever = news_vector_store.as_retriever()

    news_retriever_tool = create_retriever_tool(
        news_retriever,
        name = 'news_search',
        description='뉴스 데이터 중에서 질문과 관련된 것을 찾습니다.'
    )
    
    # agent > tool 준비
    tools = [news_retriever_tool, compare_gold_prices ,stock_analysis ,python_repl_tool]

    investment_risk_levels = {
    "위험선호형": {
        "채권": ["회사채(BB 이하)", "S&P B 이하"],
        "ELS/DLS": ["녹인 70% 이상", "녹인 130% 이하"],
        "파생결합 사채": [],
        "ELW/ETN": ["ELW", "ETN"],
        "주식": ["신용거래 관리종목", "경고종목", "위험종목"],
        "선물옵션": ["선물옵션"],
        "ETF": ["파생형"],
        "펀드": ["1등급"],
        "RP": []
    },
    "적극 투자자형": {
        "채권": ["회사채(BBB- 이상)", "S&P BB 이상"],
        "ELS/DLS": ["녹인 70% 미만", "녹인 130% 초과"],
        "파생결합 사채": [],
        "ELW/ETN": ["손실제한 ETN"],
        "주식": ["주식"],
        "선물옵션": [],
        "ETF": ["주식형", "통화형", "상품형"],
        "펀드": ["2등급"],
        "RP": []
    },
    "위험 중립형형": {
        "채권": ["회사채(BBB0-BBB+)", "S&P BBB 이상"],
        "ELS/DLS": [],
        "파생결합 사채": [],
        "ELW/ETN": [],
        "주식": [],
        "선물옵션": [],
        "ETF": ["혼합형", "주식인덱스형"],
        "펀드": ["3등급"],
        "RP": []
    },
    "안정 중립립형": {
        "채권": ["금융채", "회사채(A- 이상)", "S&P A 이상"],
        "ELS/DLS": ["원금 80% 이상 지급형"],
        "파생결합 사채": ["ELB", "DLB"],
        "ELW/ETN": [],
        "주식": [],
        "선물옵션": [],
        "ETF": ["채권형"],
        "펀드": ["4등급"],
        "RP": []
    },
    "안정형": {
        "채권": ["국고채", "통안채", "지방채", "특수채", "S&P AA 이상"],
        "ELS/DLS": [],
        "파생결합 사채": [],
        "ELW/ETN": [],
        "주식": [],
        "선물옵션": [],
        "ETF": [],
        "펀드": ["5등급"],
        "RP": ["RP"]
    }
    }

    # 프롬포트 조정 가능합니다.
    prompt = ChatPromptTemplate.from_messages(
            # "{investment_risk_levels} is an item that can be recommended based on the user's investment preferences."
        [
        (
            "system",
            "You are a helpful assistant. "
            "client's investment property is {score}"
            "{investment_risk_levels} is an item that can be recommended based on the user's investment preferences."
            "If a user requests a recommendation that does not align with their investment preferences, the request will be declined."

            "Make sure to use the `compare_gold_prices` tool to compare returns and provide investment solutions."
            '''
            아래 정보를 기반으로 사용자의 투자 성향에 맞는 최적의 투자 전략을 제안하세요.
            1. **금 수익률**
            2. **최근 S&P 500 (또는 다른 주식 지수) 수익률**

            : **투자 성향별 전략 제안**:
            - **위험 선호형(공격적 투자자)**: 높은 수익 가능성을 중시하며 단기 변동성을 감내할 수 있는 전략을 추천하세요.
            - **적극 투자형**: 주식과 금을 균형 있게 조합하여 성장성과 안정성을 모두 고려한 전략을 제안하세요.
            - **위험 중립형**: 중장기적인 성장 잠재력이 높은 시장을 중심으로 조언하세요.
            - **안정 중립형**: 안전한 투자 옵션과 함께 일부 리스크를 감수할 수 있는 조언을 제공하세요.
            - **안정형**: 원금 손실 가능성을 최소화하고, 금 투자 비중을 늘릴 수 있는 보수적인 전략을 제시하세요.
            
            투자 성향과 시장 데이터에 맞추어 현실적인 투자 조언을 작성하세요.
            그리고 조언을 한 후에 다시 매매/매도/잔고 같은 키워드를 입력하여 다시 말하도록 유도하세요.
            출력은 HTML형식으로 헤더, p 태그 등을 활용하여 만들어줘.
            '''
            ,
        ),
        
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
            #         gent_scratchpad는 에이전트(Agent)가 사고하고 계획하는 과정에서 중간 결과를 저장하는 공간입니다.
            #           특히 "실행할 도구(tool) 선택", "이미 실행한 도구의 결과를 저장" 등의 역할을 합니다.
        ]

    )

    # Agent 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

        # AgentExecutor 생성
    agent_executor = AgentExecutor(
        agent=agent,                  # 1) 사용할 에이전트
        tools=tools,                  # 2) 사용할 도구 목록
        verbose=True,                  # 3) 디버깅 메시지 출력 여부
        max_iterations=10,             # 4) 최대 실행 반복 횟수
        max_execution_time=100,         # 5) 최대 실행 시간 제한
        handle_parsing_errors=True,    # 6) 파싱 오류 처리 여부
    )
    # AgentExecutor 실행
    result = agent_executor.invoke({'score':score,'investment_risk_levels':investment_risk_levels,"input": input_string})
    messages.append({"role": "user", "content": input_string})  # 새 메시지 추가
    messages.append({"role": "assistant", "content": result['output']}) # 생성된 대답 추가

    conversations.update_one(
        {"user_id": user_id},
        {"$set": {"messages": messages}}
    )
    print('투자 권유')
    return result['output']

@csrf_exempt
def make_and_save_report(user_id,input_string):

    llm = ChatOpenAI(
        model_name='gpt-4o',
        api_key=api_key,

    )

    # mongoDB에서 데이터를 가져오기
    cluster = MongoClient(os.environ.get("mongo")) 
    db = cluster['userinfo'] # 유저 정보
    reports = db['Reports'] # 
    conversation = db['conversations']
    messages = conversation.find_one({"user_id": user_id})

    if reports.find_one({"user_id": user_id}) == None:
        reports.insert_one({"user_id": user_id,'report':[]})  # 새로운 유저 데이터 추가
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # 백터db 로드
    # NewsData.faiss
    news_vector_store = FAISS.load_local(rf'{Path(__file__).parents[0]}/vecterDB', embeddings=embeddings, allow_dangerous_deserialization=True,index_name = 'NewsData')
    # retriever 생성
    # 유사도 검색을 코사인 유사도가 아닌 다른 방법으로 하고 싶다면 as.retriever의 인자를 바꾸면 된다.
    news_retriever = news_vector_store.as_retriever()

    news_retriever_tool = create_retriever_tool(
        news_retriever,
        name = 'news_search',
        description='뉴스 데이터중에서 질문과 관련된 것을 찾습니다.'
    )
    
    # agent > tool 준비
    tools = [news_retriever_tool, compare_gold_prices ,stock_analysis ,python_repl_tool]

    # 프롬포트 조정 가능합니다.
    prompt = ChatPromptTemplate.from_messages(

        [
        (
            "system",
            '''
            너는 신문기사 데이터와 회사 공시정보를 바탕으로 물음에 정확하게 답변하는 능숙한 애널리스트이다. 
            공신력 있는 보고서의 형태로 작성해줘.
            출력은 HTML형식으로 header, p 태그 등을 활용하여 만들어줘.
            가능한 모든 툴을 사용해서 모든 정보를 포함하도록 해줘.

                1. 언제기준으로 작성했는지
                2. 정보 출처
                3. 현재가격
                4. 추세
                성장주 → PSR, PEG, DCF 활용
                가치주 → PER, PBR 활용
                자산 기반 기업(은행, 보험) → PBR, ROE 기반 평가
                부채가 많은 기업 → EV/EBITDA 활용
                5. 시가총액
                6. 거래소
                7. 일, 주, 월 퍼포먼스
                8. story : 최근정보 요약
                - 호재와 악재
                - 시장 동향
                - 변동성 수치와 왜 커졌는지 확인
                - 왜 위험한지 / 예상되는 위험이 무엇인지?
                - 실시간 뉴스
                9. 주요 코멘트
                10. 기간동안 금 수익률과 s&p 500 투자 수익률
                '''
            ,
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Agent 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

        # AgentExecutor 생성
    agent_executor = AgentExecutor(
        agent=agent,                  # 1) 사용할 에이전트
        tools=tools,                  # 2) 사용할 도구 목록
        verbose=True,                  # 3) 디버깅 메시지 출력 여부
        max_iterations=10,             # 4) 최대 실행 반복 횟수
        max_execution_time=100,         # 5) 최대 실행 시간 제한
        handle_parsing_errors=True,    # 6) 파싱 오류 처리 여부
    )
    # AgentExecutor 실행
    result = agent_executor.invoke({"input": input_string})
    
    reports.update_one(
    {"user_id": user_id},
    {"$push": {"reports": result['output']}})
    
    print('리포트 작성')

    return (result['output'])

# @csrf_exempt
# def render_markdown(text):
#     html = markdown.markdown(text)
#     return mark_safe(html)  # XSS 방지를 위해 mark_safe 처리

# 도구 생성
@tool
def compare_gold_prices(year_range: str):
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

@tool
def stock_analysis(ticker: str) -> str:
    """
    주어진 주식 티커에 대한 업데이트된 종합적인 재무 분석을 수행합니다.
    최신 주가 정보, 재무 지표, 성장률, 밸류에이션 및 주요 비율을 제공합니다.
    가장 최근 영업일 기준의 데이터를 사용합니다.
    :param ticker: 분석할 주식의 티커 심볼
    """
    def format_number(number):
        if number is None or pd.isna(number):
            return "N/A"
        return f"{number:,.0f}"
    def format_financial_summary(financials):
        summary = {}
        for date, data in financials.items():
            date_str = date.strftime('%Y-%m-%d')
            summary[date_str] = {
                "총수익": format_number(data.get('TotalRevenue')),
                "영업이익": format_number(data.get('OperatingIncome')),
                "순이익": format_number(data.get('NetIncome')),
                "EBITDA": format_number(data.get('EBITDA')),
                "EPS(희석)": f"${data.get('DilutedEPS'):.2f}" if pd.notna(data.get('DilutedEPS')) else "N/A"
            }
        return summary
    ticker = yf.Ticker(ticker)
    historical_prices = ticker.history(period='5d', interval='1d')
    last_5_days_close = historical_prices['Close'].tail(5)
    last_5_days_close_dict = {date.strftime('%Y-%m-%d'): price for date, price in last_5_days_close.items()}
    # 연간 및 분기별 재무제표 데이터 가져오기
    annual_financials = ticker.get_financials()
    quarterly_financials = ticker.get_financials(freq="quarterly")
    return str({
        "최근 5일간 종가": last_5_days_close_dict,
        "연간 재무제표 요약": format_financial_summary(annual_financials),
        "분기별 재무제표 요약": format_financial_summary(quarterly_financials),
    })

########################################################################

###### 자동 주식 매매 툴 ################################

################################
@csrf_exempt
def trading(request,user_id,name,quantity,type):
    # name : 종목 , quentity : 수량, type : 'BUY' or 'SELL'
    db = cluster['userinfo']
    user_info = db['info']
    sellect_info = user_info.find_one({"username": user_id})

    app_key = sellect_info['mock_app_key']
    app_secret = sellect_info['mock_app_secret']
    account_number = sellect_info['거래소ID']
    quantity = int(quantity)
    answer = place_order(app_key, app_secret, account_number, name, type ,quantity)
    
    return JsonResponse({'content':answer})

@csrf_exempt
def check_account(request, user_id):
    # name : 종목 , quentity : 수량, type : 'BUY' or 'SELL'
    db = cluster['userinfo']
    user_info = db['info']
    sellect_info = user_info.find_one({"username": user_id})

    app_key = sellect_info['mock_app_key']
    app_secret = sellect_info['mock_app_secret']
    account_number = sellect_info['거래소ID']

    answer = get_account_balance(app_key, app_secret, account_number)
    letter = ''
    for i,j in answer.items():
        k = f'<tr><td>{i}</td><td>{j}</td></tr>'
        letter += k
    
    final_letter = '<thead><tr><th>항목</th><th>값</th></tr></thead><tbody>'+ letter +'</tbody>'
    return JsonResponse({'content':final_letter})

#################################################################

@csrf_exempt
def get_stock_price(app_key, app_secret, ticker):
    # :작은_파란색_다이아몬드: API 기본 URL (모의투자 환경)
    url_base = "https://openapivts.koreainvestment.com:29443"
    # :작은_파란색_다이아몬드: 1. Access Token 발급
    headers = {"content-type": "application/json"}
    path = "oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret
    }
    url = f"{url_base}/{path}"
    res = requests.post(url, headers=headers, data=json.dumps(body))
    access_token = res.json().get('access_token', None)
    if not access_token:
        print(" Access Token 발급 실패! 응답:", res.text)
        return
    print(":Access Token:", access_token)
    # :작은_파란색_다이아몬드: 2. 해외주식 현재가 조회 API 호출
    path = "uapi/overseas-price/v1/quotations/price"  # :URL 수정됨
    url = f"{url_base}/{path}"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "HHDFS00000300"  # :올바른 tr_id 사용
    }
    # :작은_파란색_다이아몬드: 종목 코드 설정
    params = {"AUTH": "", "EXCD": "NAS", "SYMB": ticker}  # :선택된 종목 코드
    print("\n:작은_파란색_다이아몬드: 조회 중:", params["SYMB"], "거래소:", params["EXCD"])
    res = requests.get(url, headers=headers, params=params)
    print("API 응답 상태 코드:", res.status_code)
    print("API 응답 본문:", res.text)
    if res.status_code == 200:
        try:
            stock_data = res.json()
            if stock_data["rt_cd"] == "0":
                return(stock_data['output']['last'])
            else:
                return
        except json.JSONDecodeError:
            return
    else:
        return

# 계좌 잔고 확인 코드
@csrf_exempt
def get_account_balance(app_key, app_secret, account_number):
    # :작은_파란색_다이아몬드: API 기본 URL (모의투자 환경)
    url_base = "https://openapivts.koreainvestment.com:29443"
    # :작은_파란색_다이아몬드: 1. Access Token 발급
    headers = {"content-type": "application/json"}
    path = "oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret
    }
    url = f"{url_base}/{path}"
    res = requests.post(url, headers=headers, data=json.dumps(body))
    access_token = res.json().get('access_token', None)
    if not access_token:
        print(" Access Token 발급 실패! 응답:", res.text)
        return
    print(":Access Token:", access_token)
    # :작은_파란색_다이아몬드: 2. 해외주식 잔고 조회 API 호출
    path = "uapi/overseas-stock/v1/trading/inquire-balance"
    url = f"{url_base}/{path}"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTS3012R"  # :모의투자 거래 ID
    }
    # :작은_파란색_다이아몬드: 계좌번호 정보 (모의투자 해외주식 계좌번호)
    params = {
        "CANO": account_number[:8],  # :계좌번호 앞 8자리
        "ACNT_PRDT_CD": account_number[8:],  # :계좌 상품 코드 (2자리)
        "OVRS_EXCG_CD": "NASD",  # :해외거래소 코드 (나스닥)
        "TR_CRCY_CD": "USD",     # :거래 통화 코드 (미국 달러)
        "CTX_AREA_FK200": "",    # :최초 조회 (연속조회 없음)
        "CTX_AREA_NK200": ""     # :최초 조회 (연속조회 없음)
    }
    res = requests.get(url, headers=headers, params=params)
    # :작은_파란색_다이아몬드: 3. API 응답 데이터 확인
    print("API 응답 상태 코드:", res.status_code)
    if res.status_code == 200:
        try:
            stock_data = res.json()
            if stock_data["rt_cd"] == "0":
                output = stock_data.get('output1', [])
                if not output:
                    return  (":작은_파란색_다이아몬드: 잔고가 없습니다.")
                else:
                    for item in output:
                        return ({"종목명:": item.get('ovrs_item_name', "N/A"),
                        "보유 수량:": item.get('ovrs_cblc_qty', "N/A"),
                        "매도 가능 수량:": item.get('ord_psbl_qty', "N/A"),
                        "매입 평균 가격:": item.get('pchs_avg_pric', "N/A"),
                        "현재 가격:": item.get('now_pric2', "N/A"),
                        "평가 손익:": item.get('frcr_evlu_pfls_amt', "N/A"),
                        "평가 손익률:": item.get('evlu_pfls_rt', "N/A"),
                        "해외주식 평가금액:": item.get('ovrs_stck_evlu_amt', "N/A")})
            else:
                return (" API 요청 실패! 응답 메시지:")
        except json.JSONDecodeError:
            return ("JSON 파싱 오류! 응답 본문:")
    else:
        return (" API 요청 실패:")

# 주식 매매 함수
@csrf_exempt
def place_order(app_key, app_secret, account_number, ticker, order_type, order_qty):
    # ticker 티커 order_type: "BUY" or "SELL" order_qty: 주문 수량 order_price: 주문 가격
    # :작은_파란색_다이아몬드: API 기본 URL (모의투자 환경)
    url_base = "https://openapivts.koreainvestment.com:29443"
    # :작은_파란색_다이아몬드: 1. Access Token 발급
    headers = {"content-type": "application/json"}
    path = "oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret
    }
    url = f"{url_base}/{path}"
    res = requests.post(url, headers=headers, data=json.dumps(body))
    access_token = res.json().get('access_token', None)
    if not access_token:
        return JsonResponse({'content':" Access Token 발급 실패! 응답:"})
    print(":Access Token:", access_token)
    # :작은_파란색_다이아몬드: 주문 유형에 따른 거래 ID (모의투자 기준)
    if order_type.upper() == "BUY":
        tr_id = "VTTT1002U"  # 미국 주식 매수 (모의투자)
    elif order_type.upper() == "SELL":
        tr_id = "VTTT1001U"  # 미국 주식 매도 (모의투자)
    else:
        print(" 잘못된 주문 유형입니다. 'BUY' 또는 'SELL'만 사용하세요.")
        return JsonResponse({'content':""" 잘못된 주문 유형입니다. 'BUY' 또는 'SELL'만 사용하세요."""})
    # :작은_파란색_다이아몬드: API 경로 설정
    path = "uapi/overseas-stock/v1/trading/order"
    url = f"{url_base}/{path}"
    # :작은_파란색_다이아몬드: 요청 데이터 설정
    order_price = get_stock_price(app_key, app_secret, ticker)
    data = {
        "CANO": account_number[:8],  # :계좌번호 앞 8자리
        "ACNT_PRDT_CD": account_number[8:],  # :계좌 상품 코드 (2자리)
        "OVRS_EXCG_CD": "NASD",  # :해외 거래소 코드 (나스닥)
        "PDNO": ticker,  # :종목 코드
        "ORD_QTY": str(order_qty),  # :주문 수량
        "OVRS_ORD_UNPR": str(order_price),  # :주문 단가 (지정가 주문 필요)
        "CTAC_TLNO": "",  # 연락처 (옵션)
        "MGCO_APTM_ODNO": "",  # 운용사 지정 주문번호 (옵션)
        "ORD_SVR_DVSN_CD": "0",  # :주문 서버 구분 코드
        "ORD_DVSN": "00"  # :주문 구분 (00: 지정가 주문)
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
        "custtype": "P"
    }
    # :작은_파란색_다이아몬드: API 요청 (POST)
    res = requests.post(url, headers=headers, data=json.dumps(data))
    # :작은_파란색_다이아몬드: 응답 확인
    if res.status_code == 200:
        response_data = res.json()
        if response_data.get("rt_cd") == "0":
            return ('결재가 완료 되었습니다!')
        else:
            return ('주문 실패!')
    else:
        return ('API 요청 실패!')






