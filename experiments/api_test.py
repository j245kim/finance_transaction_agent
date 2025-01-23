import os

from dotenv import load_dotenv


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


from langchain_core.prompts import PromptTemplate

# template 정의
template = "{country}의 수도는 어디인가요?"

# from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt_template = PromptTemplate.from_template(template)
prompt_template

# prompt 생성
prompt = prompt_template.format(country="대한민국")
prompt

# prompt 생성
prompt = prompt_template.format(country="미국")
prompt

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)