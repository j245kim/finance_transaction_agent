import atexit
from pathlib import Path


from llama_cpp import Llama
from transformers import AutoTokenizer


@atexit.register
def free_model():
    model._sampler.close()
    model.close()


model_dir_path = rf'{Path(__file__).parent}\model'
model_path = rf'{model_dir_path}\llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf'
tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
model = Llama(model_path)

instruction = "철수가 20개의 연필을 가지고 있었는데 영희가 절반을 가져가고 민수가 남은 5개를 가져갔으면 철수에게 남은 연필의 갯수는 몇개인가요?"

messages = [
                {
                    'role': 'system',
                    "content": "You are a very smart AI chatbot. Please answer user questions accurately and kindly. 당신은 아주 똑똑한 AI 챗봇입니다. 사용자의 질문에 정확하고, 친절하게 답변해주세요."
                }, 
                {
                    "role": "user",
                    "content": f"{instruction}"
                }
            ]

prompt = tokenizer.apply_chat_template(
                                            messages, 
                                            tokenize = False,
                                            add_generation_prompt=True
                                        )

prompt = prompt.replace('<|begin_of_text|>', '').replace('<|eot_id|>', '')
prompt = prompt.replace('<|start_header_id|>', '\n\n<|start_header_id|>').strip()

generation_kwargs = {
                        "max_tokens":512,
                        "stop":["<|eot_id|>"],
                        "echo":True,
                        "top_p":0.9,
                        "temperature":0.6,
                    }

resonse_msg = model(prompt, **generation_kwargs)
# print(resonse_msg['choices'][0]['text'])
print(resonse_msg['choices'][0]['text'][len(prompt):].strip())