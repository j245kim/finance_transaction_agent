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
from datetime import datetime

from django.shortcuts import render
from django.http import JsonResponse, FileResponse

from pymongo import MongoClient

cluster = MongoClient(os.environ.get("mongo")) # 클러스터 
db = cluster['userinfo'] # 유저 정보
collection = db["info"] # 내가 지정한 컬렉션 이름

# Create your views here.

def load_images(req, image_name):
    image_path = rf'{Path(__file__).parents[1]}\images'
    match image_name:
        case 'downarrow':
            image_path = rf'{image_path}\model_images\down_arrow.png'
        case 'bllossom':
            image_path = rf'{image_path}\model_images\bllossom_mini_icon.png'
        case 'chatgpt':
            image_path = rf'{image_path}\model_images\chatgpt.png'
        case 'huggingface':
            image_path = rf'{image_path}\model_images\hf_icon.png'
        case 'check':
            image_path = rf'{image_path}\model_images\check.png'
        case _:
            image_path = rf'{image_path}\model_images\question_mark.png'
        
    return FileResponse(open(image_path, 'rb'), content_type='image/png')

def bllossom(req, message):
    def prompt_gen(tokenizer: PreTrainedTokenizer, messages: list[dict[str, str]], max_tokens) -> str:
        """사전훈련 된 Tokenizer와 messages를 통해 prompt를 생성하는 함수
        
        Args:
            tokenizer (PreTrainedTokenizer): 사전훈련 된 Tokenizer
            messages (list[dict[str, str]]): 대화 내용이 담긴 messages
            max_tokens (int): 생성할 prompt의 최대 길이
            
        Returns:
            prompt (str): 생성된 prompt"""
        
        prompt = tokenizer.apply_chat_template(
                                                    messages, 
                                                    tokenize=False,
                                                    add_generation_prompt=True
                                                )

        prompt = prompt.replace('<|begin_of_text|>', '').replace('<|eot_id|>', '')
        prompt = prompt.replace('<|start_header_id|>', '\n\n<|start_header_id|>').strip()

        if len(prompt) >= max_tokens:
            del messages[1:3]
            prompt = prompt_gen(tokenizer, messages, max_tokens)

        return prompt
    
    model_dir_path = rf'{Path(__file__).parents[1]}\models\Bllossom'
    model_path = rf'{model_dir_path}\llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf'
    tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
    model = Llama(model_path)
    # session 디렉토리에 있는 message.json 파일 경로
    session_path = rf'{Path(__file__).parents[1]}\session\messages.json'

    # Messages를 가져온 다음, 현재 요청한 message 추가
    with open(session_path, encoding='utf-8', errors='ignore') as f:
        messages = json.load(f)
    messages.append({"role": "user", "content": message})
    
    # ChatBot 대답
    max_tokens = 1024
    prompt = prompt_gen(tokenizer, messages, max_tokens)

    generation_kwargs = {
                            "max_tokens": max_tokens,
                            "stop": ["<|eot_id|>"],
                            "echo": True,
                            "top_p": 0.9,
                            "temperature": 0.6,
                        }

    resonse_msg = model(prompt, **generation_kwargs)
    completion = resonse_msg['choices'][0]['text'][len(prompt):].strip()

    model._sampler.close()
    model.close()

    messages.append({"role": "assistant", "content": completion})

    with open(session_path, mode='w', encoding='utf-8', errors='ignore') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

    return JsonResponse({'content': completion}, json_dumps_params={'ensure_ascii': False}, safe=False, status=200)

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

    if conversations.find_one({"user_id": user_id}) == None:
        conversations.insert_one({"user_id": user_id, "messages": []})  # 새로운 유저 데이터 추가


    conversation = conversations.find_one({"user_id": user_id})

    messages = conversation["messages"]  # 기존 메시지 리스트 가져오기

    messages.append({"role": "user", "content": input_string})  # 새 메시지 추가

    prompt = PromptTemplate.from_template('당신은 아주 똑똑한 AI 챗봇입니다. 사용자의 질문에 정확하고, 친절하게 답변해주세요. 대화 기록 = {message} 질문 = {content}')
    # 페르소나 조정 가능합니다.

    chain = prompt | llm

    answer = chain.invoke(input={"message": messages, "content": input_string})

    messages.append({"role": "assistant", "content": answer.content})

    conversations.update_one(
        {"user_id": user_id},
        {"$set": {"messages": messages}}
    )
    
    return JsonResponse({'content':answer.content})

def etc(req, message):
    return JsonResponse({'content': '죄송합니다. 해당 모델은 아직 구현되지 않아서 답변을 할 수 없습니다.'}, json_dumps_params={'ensure_ascii': False}, safe=False, status=200)

def invest_chat(req, invest_rank):
    return JsonResponse({'content': '확인'}, json_dumps_params={'ensure_ascii': False}, safe=False, status=200)

def classification_finance(request, input_string):
# 금융 관련 질문 여부 확인
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY_sesac')
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        api_key=api_key
    )

    prompt = PromptTemplate.from_template('{message}에 대해 금융에 관련된 내용이면 True, 아니면 False를 반환하세요.')
    # 프롬포트 조정 가능합니다.
    chain = prompt | llm

    answer = chain.invoke({'message': input_string})

    return JsonResponse({'type':answer.content})

def classification_invest(request, input_string):
# 투자 관련 진물 여부 확인
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY_sesac')
    llm = ChatOpenAI(
        model_name='gpt-4o-mini',
        api_key=api_key
    )

    prompt = PromptTemplate.from_template('{message}에 대해 투자에 관련된 내용이면 True, 아니면 False를 반환하세요.')
    # 프롬포트 조정 가능합니다. (페르소나)
    chain = prompt | llm

    answer = chain.invoke({'message': input_string})

    return JsonResponse({'type':answer.content})

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
 




