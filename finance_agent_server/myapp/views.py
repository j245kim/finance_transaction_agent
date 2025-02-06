import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# RAG 사용을 위한 라이브러리리
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
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

from pymongo import MongoClient

# stt 구현하는 라이브러리
from django.views.decorators.csrf import csrf_exempt  # CSRF 보안 토큰 무시
from .stt import speech_to_text 
import whisper
import io

# tts 구현하는 라이브러리
from gtts import gTTS

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

    if classification_invest(input_string) == 'False':
        if classification_market_situation(input_string) == 'False':
            # 프롬포트 조정 가능합니다.
            prompt = PromptTemplate.from_template(
                '''
                당신은 투자어드바이저입니다. 
                질문에 대한 답변을 해주세요. 
                대화 기록 = {message} 질문 = {content}
                ''')
        

            chain = prompt | llm

            answer = chain.invoke(input={"message": messages, "content": input_string})

            messages.append({"role": "assistant", "content": answer.content})

            conversations.update_one(
                {"user_id": user_id},
                {"$set": {"messages": messages}}
            )
            
            return JsonResponse({'content':answer.content})
        
        else:
            # RAG 사용 >> 함수로 불러올 예정입니다.
            answer = invest_search_rag(input_string)

            return JsonResponse({'content':answer})

def classification_finance(input_string):
# 금융 관련 질문 여부 확인
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY_sesac')
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        api_key=api_key,
        temperature = 0
    )

    prompt = PromptTemplate.from_template('''
    message : '{message}'

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
        api_key=api_key,
        temperature = 0
    )

    prompt = PromptTemplate.from_template(
        '''
        message : "{message}"
        이 문장을 말한 사람이 실제로 투자를 할 의도가 있는지 여부를 판단하세요.
        투자와 관련된 내용이 있는 경우 "True"를 반환하고, 그렇지 않은 경우 "False"를 반환하세요.
        ''')
    # 프롬포트 조정 가능합니다. (페르소나)
    chain = prompt | llm

    answer = chain.invoke({'message': input_string})

    return answer.content

def classification_market_situation(input_string):
    # 시장 현황 질문 여부 확인
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY_sesac')
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        api_key=api_key,
        temperature = 0
    )

    prompt = PromptTemplate.from_template(
        '''
        message : "{message}"
        이 문장을 말한 사람이 시장 현황에 대해 물어보고 있는지 여부를 판단하세요.
        시장 현황과 관련된 내용이 있는 경우 "True"를 반환하고, 그렇지 않은 경우 "False"를 반환하세요.
        ''')
    # 프롬포트 조정 가능합니다. (페르소나)
    chain = prompt | llm

    answer = chain.invoke({'message': input_string})

    return answer.content

def invest_search_rag(input_string):
    # RAG 제작

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY_sesac')
    
    llm = ChatOpenAI(
        model_name='gpt-4o',
        api_key=api_key
    )

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
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # faiss에 저장
    if not os.path.isfile(rf'{Path(__file__).parent}/vectorDB/NewsData.faiss') :
        vector_store = FAISS.from_documents(split_texts, embeddings)
    
        vector_store.save_local(Path(__file__).parent/'vectorDB',index_name = 'NewsData')

    vector_store = FAISS.load_local(Path(__file__).parent/'vectorDB',embeddings,allow_dangerous_deserialization=True,index_name = 'NewsData')

    # retriever 생성
    # 유사도 검색을 코사인 유사도가 아닌 다른 방법으로 하고 싶다면 as.retriever의 인자를 바꾸면 된다.
    retriever = vector_store.as_retriever()

    system_prompt = ''' 
    Use the given context to answer the question. If you don't know the answer, say you don't know.  
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
      '''
    #페르소나 조정 가능합니다.
    
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )

    # llm과 prompt 연결결
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 리트리버 연결결
    chain = create_retrieval_chain(retriever, question_answer_chain)

    chain.invoke({"input": input_string})

    answer = chain.invoke({"input": input_string})
    return answer['answer']





 
def stt_api(request):
    # Whisper 모델 로드
    model = whisper.load_model("tiny")

    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]

        # 파일 저장 없이 메모리에서 바로 변환
        audio_bytes = audio_file.read()
        audio_stream = io.BytesIO(audio_bytes)
        
        # Whisper 변환 실행
        result = model.transcribe(audio_stream, language="ko")  # 한국어 지정
        
        return JsonResponse({"text": result["text"]})
    
    return JsonResponse({"error": "음성 파일을 업로드하세요."}, status=400)

def tts_api(request, input_string):
    # 입력 텍스트가 비어 있는지 확인
    if not input_string.strip():
        return JsonResponse({"error": "텍스트가 비어 있습니다."}, status=400)

    # gTTS를 사용하여 음성 변환
    tts = gTTS(text=input_string, lang="ko")
    audio_stream = io.BytesIO()
    tts.write_to_fp(audio_stream)

    # 오디오 스트림을 HTTP 응답으로 반환
    audio_stream.seek(0)
    response = HttpResponse(audio_stream.read(), content_type="audio/mpeg")
    response["Content-Disposition"] = 'inline; filename="tts_audio.mp3"'
    return response


