import os
import json
from pathlib import Path

from dotenv import load_dotenv
from llama_cpp import Llama
from transformers import AutoTokenizer, PreTrainedTokenizer

from django.shortcuts import render
from django.http import JsonResponse, FileResponse

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


def chatgpt(req, message):
    load_dotenv()
    api_key = os.getenv('OPEN_API_KEY_sesac')


def invest_chat(req, invest_rank):
    return JsonResponse({'response': '확인'}, json_dumps_params={'ensure_ascii': False}, safe=False, status=200)