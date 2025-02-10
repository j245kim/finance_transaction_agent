import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors.length_based import LengthBasedExampleSelector

examples = [
    {"food": "Kimchi", "category": "Korean food"},
    {"food": "Chocolate", "category": "dessert"},
    {"food": "Pasta", "category": "Italian food"},
    {"food": "Americano", "category": "Coffee"},
]

example_prompt = PromptTemplate(
    template="Food:{food} Category:{category}", input_variables=["food", "category"]
)
example_selector = LengthBasedExampleSelector(
    examples=examples, example_prompt=example_prompt, max_length=30
)
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="What is the category of this food?",
    suffix="Food: {food}",
    input_variables=["food"],
)

output = dynamic_prompt.format(food="Korean BBQ")
print(len(output.split()), output)