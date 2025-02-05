from pymongo import MongoClient

from dotenv import load_dotenv
import os 

# load .env
load_dotenv()


cluster = MongoClient(os.environ.get("mongo"))
db = cluster['userinfo']
collection = db["info"]

my_info = {
    "name": "ronaldo",
    "age": "60",
}

collection.insert_one(my_info)

for result in collection.find({}):
    print(result)