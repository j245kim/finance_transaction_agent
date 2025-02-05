import os
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter


from dotenv import load_dotenv
from llama_cpp import Llama
from transformers import AutoTokenizer, PreTrainedTokenizer

from pymongo import MongoClient
from datetime import datetime, timedelta

from django.shortcuts import render
from django.http import JsonResponse, FileResponse

from pymongo import MongoClient

cluster = MongoClient(os.environ.get("mongo")) # 클러스터 
db = cluster['userinfo'] # 유저 정보
collection = db["info"] # 내가 지정한 컬렉션 이름

# Create your views here.




def chatgpt(req, user_id ,input_string):

    load_dotenv()

    cluster = MongoClient(os.environ.get("mongo")) # 클러스터 
    db = cluster['userinfo'] # 유저 정보
    conversations = db["conversations"] # 내가 지정한 컬렉션 이름

    api_key = os.getenv('OPENAI_API_KEY_sesac')

    llm = ChatOpenAI(
        model_name='gpt-4o',
        max_tokens = 50,
        api_key=api_key
    )
    user = conversations.find_one({"user_id": user_id})

    if conversations.find_one({"user_id": user_id}) == None:
        conversations.insert_one({"user_id": user_id,'stack': [], 'ban':[] , "messages": []})  # 새로운 유저 데이터 추가
 
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

    prompt = PromptTemplate.from_template(
        '''
        당신은 투자어드바이저입니다. 
        사용자의 질문을 받아들이고, 투자에 관련된 질문이 있는지 확인하세요.
        classification_invest() 함수는 질문을 string으로 받고, 관계가 있다면 True를 반환하고, 그렇지 않다면 False를 반환합니다.
        classification_invest() 함수를 사용하여 사용자의 질문이 투자와 관련이 있는지 확인하고,
        관계가 있다면 투자 성향 체크를 진행하고
        관계가 없다면 질문에 대한 대답을 해주세요.

        대화 기록 = {message} 질문 = {content}
        ''')
    # 페르소나 조정 가능합니다.

    chain = prompt | llm

    answer = chain.invoke(input={"message": messages, "content": input_string})

    messages.append({"role": "assistant", "content": answer.content})

    conversations.update_one(
        {"user_id": user_id},
        {"$set": {"messages": messages}}
    )
    
    return JsonResponse({'content':answer.content})

def classification_finance(input_string):
# 금융 관련 질문 여부 확인
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY_sesac')
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        api_key=api_key
    )

    prompt = PromptTemplate.from_template('''
    {message}

    금융과 무관한 내용을 금융과 억지로 연결 짓지 마세요. 
    금융(예: 주식, 투자, 은행, 대출, 경제 등)과 직접 관련이 있는 경우에만 "True"를 반환하고, 그렇지 않다면 "False"를 반환하세요.
    답변은 "True" 또는 "False"만 출력하세요.
    ''')
    # 프롬포트 조정 가능합니다.
    
    chain = prompt | llm

    answer = chain.invoke({'message': input_string})

    return answer.content

def classification_invest(input_string):
# 투자 관련 진물 여부 확인
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY_sesac')
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        api_key=api_key
    )

    prompt = PromptTemplate.from_template(
        '''
        "{message}"
        이 문장을 말한 사람이 실제로 투자를 할 의도가 있는지 여부를 판단하세요.
        투자와 관련된 내용이 있는 경우 "True"를 반환하고, 그렇지 않은 경우 "False"를 반환하세요.
        ''')
    # 프롬포트 조정 가능합니다. (페르소나)
    chain = prompt | llm

    answer = chain.invoke({'message': input_string})

    return answer.content

def invest_search_rag(request, input_string):
    # RAG 제작

    load_dotenv()
    api_key = os.getenv('OPEN_API_KEY_sesac')
    
    llm = ChatOpenAI(
        model_name='gpt-4o',
        api_key=api_key
    )

    prompt = PromptTemplate.from_template('')
    #페르소나 조정 가능합니다.

    client = MongoClient('')
    # Atlas 사용 시에 연결 주소 입력

    db = client['']
    # 데이터베이스 이름 입력

    collection = db['']
    # 컬렉션 이름 입력  

    data = collection.find()

    chain = prompt | llm

    answer = chain.invoke({'message': input_string})

    return JsonResponse({'report':answer.content})

def make_report(requeset, input_string):

    load_dotenv()
    api_key = os.getenv('OPEN_API_KEY_sesac')

    # retriever = 
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        retriever = retriever,
        api_key=api_key
    )

    prompt = PromptTemplate.from_template('')

    chain = prompt | llm

    answer = chain.invoke({'message': input_string})

    return JsonResponse({'report':answer.content})

def RUB(request, input_string):

    # load_dotenv()
    # api_key = os.getenv('OPEN_API_KEY_sesac')
    # llm = ChatOpenAI(
    #     model_name='gpt-4o-mini',
    #     api_key=api_key
    # )
    


    return JsonResponse({})
 




