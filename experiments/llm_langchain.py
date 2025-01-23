# 파이썬 표준 라이브러리
from pathlib import Path

# 파이썬 서드파티 라이브러리
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate


model_path = rf'{Path(__file__).parents[0]}\model\llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf'

llm = ChatLlamaCpp(
                    temperature=0.6,
                    model_path=model_path,
                    max_tokens=512,
                    repeat_penalty=1.5,
                    top_p=0.9,
                    verbose=True,
                    echo=True,
                    stop=["<|eot_id|>"]
                )

# messages = [
#                 (
#                     "system",
#                     "You are a very smart AI chatbot. Please answer user questions accurately and kindly. 당신은 아주 똑똑한 AI 챗봇입니다. 사용자의 질문에 정확하고, 친절하게 답변해주세요.",
#                 ),
#                 (
#                     "human",
#                     "철수가 20개의 연필을 가지고 있었는데 영희가 절반을 가져가고 민수가 남은 5개를 가져갔으면 철수에게 남은 연필의 갯수는 몇개인가요?"
#                 ),
#             ]

# ai_msg = llm.invoke(messages)
# print(ai_msg.content)

prompt = ChatPromptTemplate.from_messages(
                                            [
                                                (
                                                    "system",
                                                    "당신은 {field}에 아주 똑똑한 AI 챗봇입니다. 사용자의 질문에 정확하고, 친절하게 답변해주세요.",
                                                ),
                                                (
                                                    "human",
                                                    "{input}"
                                                ),
                                            ]
                                        )

chain = prompt | llm
ai_msg = chain.invoke(
                        {
                            "field": "논리",
                            "input": "철수가 20개의 연필을 가지고 있었는데 영희가 절반을 가져가고 민수가 남은 5개를 가져갔으면 철수에게 남은 연필의 갯수는 몇개인가요?",
                        }
                    )
print(prompt)
# print(ai_msg.content)