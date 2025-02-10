import os
from pathlib import Path

from dotenv import load_dotenv

from django.shortcuts import render
from django.http import JsonResponse, FileResponse, StreamingHttpResponse

from langchain_openai import ChatOpenAI

# Create your views here.

def home(req):
    return render(req, r'myapp/chat_page.html')


def load_images(req, image_name):
    image_path = rf'{Path(__file__).parents[1]}/images'
    match image_name:
        case 'downarrow':
            image_path = rf'{image_path}/model_images/down_arrow.png'
        case 'bllossom':
            image_path = rf'{image_path}/model_images/bllossom_mini_icon.png'
        case 'chatgpt':
            image_path = rf'{image_path}/model_images/chatgpt.png'
        case 'huggingface':
            image_path = rf'{image_path}/model_images/hf_icon.png'
        case 'check':
            image_path = rf'{image_path}/model_images/check.png'
        case _:
            image_path = rf'{image_path}/model_images/question_mark.png'
        
    return FileResponse(open(image_path, 'rb'), content_type='image/png')
    

def bllossom(req, message):
    return JsonResponse({'content': '죄송합니다. Bllossom 모델은 아직 구현되지 않아서 답변을 할 수 없습니다.'}, json_dumps_params={'ensure_ascii': False}, safe=False, status=200)


def chatgpt(req, message):
    def streaming(message):
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')

        llm = ChatOpenAI(model_name='gpt-4o-mini', api_key=api_key)

        response = llm.stream(message)

        for chunk in response:
            yield chunk

    return JsonResponse({'content': '죄송합니다. GPT 모델은 아직 구현되지 않아서 답변을 할 수 없습니다.'}, json_dumps_params={'ensure_ascii': False}, safe=False, status=200)
    return StreamingHttpResponse(streaming(message), content_type="text/event-stream")


def etc(req, message):
    return JsonResponse({'content': '죄송합니다. 해당 모델은 아직 구현되지 않아서 답변을 할 수 없습니다.'}, json_dumps_params={'ensure_ascii': False}, safe=False, status=200)


def invest_chat(req, invest_rank):
    return JsonResponse({'content': '미구현'}, json_dumps_params={'ensure_ascii': False}, safe=False, status=200)