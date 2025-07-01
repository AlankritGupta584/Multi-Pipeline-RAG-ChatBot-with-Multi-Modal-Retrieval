# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import requests
import json
import random
import time

import os
import google.generativeai as genai
import os
import random
import google.generativeai as genai

# print(1)
api_keys = [
    "please use your own api key"
]
# print(2)
def get_random_model():
    api_key = random.choice(api_keys)
    if(api_key=="please use your own api key"):
        print("No valid API found")
        raise ValueError("Missing API key: Please use your own Gemini API key instead of the placeholder. Please check their docs for more.")

    genai.configure(api_key=api_key)

    return genai.GenerativeModel("models/gemini-1.5-flash")


# models=genai.list_models()
# for m in models:
#     print(m)
model_emb = SentenceTransformer("nomic-ai/modernbert-embed-base")

def generate_embeddings(text):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    em = model_emb.encode(texts, reference_compile=False)
    
    if isinstance(text, str):
        return em[0]
    
    
print(1)
API_KEY="please use your own api key"
if(API_KEY=="please use your own api key"):
        print("No valid API found")
        raise ValueError("Missing API key: Please use your own Pinecone API key instead of the placeholder. Please check their docs for more.")

pc = Pinecone(api_key=API_KEY)

index_name = "quickstart"
index = pc.Index(index_name)


def semantic_search2(query, k=10):
    ans=[]
    query_e=generate_embeddings(query)
    # print("Please type k, try to keep around 10 or 15. More k means more accuracy, but slow confused result. Less k means less accuracy, but faster.")
    try:
        # k=int(input("Number of top k paragraphs you want (out of 30): "))
        k=10
    except:
        print("Non valid value, k = 10 for results.")
        k=10
    # print("\n")
    print("Your chosen k is: ", k)
    results = index.query(
        vector=query_e.tolist(),
        top_k=k,
        include_metadata=True
    )
    
    for match in results.matches:
        ans.append(match["metadata"]["text"])
    ans = list(set(ans))

    return ans


# print(semantic_search2("PLD temperature for ZnO"))

# API_OPENROUTER="HIDDEN FOR PRIVACY"

# class ResponseWrapper:
#     def __init__(self, text):
#         self.text = text

# class Model:
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.api_url = "https://openrouter.ai/api/v1/chat/completions"
#         self.model_name = "google/gemini-flash-1.5-8b"
#         self.headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#             # Optional:
#             # "HTTP-Referer": "<YOUR_SITE_URL>",
#             # "X-Title": "<YOUR_SITE_NAME>"
#         }

#     def generate_content(self, prompt):
#         data = {
#             "model": self.model_name,
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             "max_tokens": 100  
#         }

#         response = requests.post(
#             url=self.api_url,
#             headers=self.headers,
#             data=json.dumps(data)
#         )

#         if response.status_code == 200:
#             result = response.json()
#             text = result['choices'][0]['message']['content']
#             return ResponseWrapper(text)
#         else:
#             error_msg = f"Error {response.status_code}: {response.text}"
#             return ResponseWrapper(error_msg)

# API_OPENROUTER="HIDDEN FOR PRIVACY"

# model = Model(api_key=API_OPENROUTER)
# r = model.generate_content("How are you")
# print(r.text)


print(2)
def genfirst(query, summ):
    prompt = f"""
You are given query. Find the most appropriate one or multiple paragraphs.
Also tell a little about each paragraph by explicitly mentioning the asked parameter in query.

Query: {query}


"""
    # print("Summarized paras: ")
    # print(summ)

    for i, para in enumerate(summ, 1):
        prompt += f"Paragraph {i}: {para}\n"
    # r=ollama(prompt)
    retries = 3 
    success = False
    
    while retries > 0 and not success:
        try:
            model = get_random_model()
            r = model.generate_content(prompt)
            success = True
            
        except Exception as e:
            retries -= 1
            if retries > 0:
                time.sleep(3)

    r=r.text
    return r



waiting_messages = [
    "Evaporating irrelevant data — condensing only what matters...",
    "Hold tight! Just ablating some knowledge from the right source.",
    "Target locked. Pulsing through dense research material...",
    "Synthesizing your solution — atom by atom.",
    "Filtering noise, amplifying the signal — your answer is forming.",
    "Gathering atoms of info… precision takes a second!",
    "Tuning parameters… the perfect reply is on its way.",
    "Just refining a few layers of thought…",
    "Your answer is almost deposited in the chamber.",
    "Reconstructing knowledge — coherence in progress.",
    "Bouncing off a few ideas… your insight is seconds away.",
    "We apologise for the delay, our bot was having a nap.",
    "Apologies for the wait! We’re just bribing the servers with coffee.",
    "Thanks for your patience!",
    "Your data is packed in a box… and the delivery guy will deliver it to you soon.",
    "Hang tight, we're getting that info for you...",
    "Just a moment — gathering the best response.",
    "Working on it… almost there!",
    "Give us a second to fetch the right answer.",
    "One sec — making sure it's accurate.",
    "Hold on… putting the pieces together.",
    "Processing your request — thanks for your patience.",
    "Looking that up… won't be long!",
    "Just checking the facts — back in a flash.",
    "Getting everything ready for you..."
]



def querygen(query):

    prompt=f"""
    You are given a query. Return only the parameters in quotes.
    Return the answer in inverted comma.
    Example: "What is temperature of X thin films over KrF laser" then return "temperature X thin films KrF laser"

    Query: {query}

    """
    retries = 3 
    success = False
    while retries > 0 and not success:
        try:
            model = get_random_model()
            r = model.generate_content(prompt)
            success = True
        except Exception as e:
            print(e)
            retries -= 1
            if retries > 0:
                time.sleep(3)

    r=r.text

    return r

print(3)

def summarizer(query):
    print("You're into semantic search 🔎...")
    q=querygen(query)
    sim = semantic_search2(q)
    print("Analysing the retrieved paragraphs 🕵🏻...")
    # print("...")
    
    summ = []
    for para in sim:
        # scheck=same_checker(query, para)
        # if "yes" in scheck.lower():
        #     pass
        # else: 
        #     continue
        retries = 3 
        success = False
        
        # print("printing para")
        # print(para)

        while retries > 0 and not success:
            try:
                model = get_random_model()
                response = model.generate_content(f"""
                                                  Summarize the given text in only two lines (30 words).
                                                  Do not lose any information about parameters and materials, like temperature, wavelength, pressure, material name, substrate name, etc.
                                                  
                                                  Text:
                                                  {para}
""")
                summ.append(response.text)
                success = True
            except Exception as e:
                retries -= 1
                if retries > 0:
                    time.sleep(3)
                
    return summ

def gensecond_semantic(query):
    print("Thank you, We got your query✅...")
    # print("...")

    
    summ = summarizer(query)

    # print(summ)

    random_message = random.choice(waiting_messages)
    print(random_message)
    # print("...")

    r1 = genfirst(query, summ)

    # print(r1)
    print("Generating your answer✍🏻...")

    
    
    prompt = f"""
You have a query and the hint to the query about PLD (Pulsed Laser Deposition).

1. Present to me it like you didn't get any hint and you're speaking it using your data. DON'T MENTION ANYTHING ABOUT PARAGRAPHS EXPLICITLY.
2. Present every information. Dont miss anything, you have to present it in 30 to 150 words maximum, unless query asks to make it bigger.

Query: 
{query}


Hint: 
{r1}

"""

    print("--" * 60)
    retries = 3 
    success = False
    
    while retries > 0 and not success:
        try:
            model = get_random_model()
            r = model.generate_content(prompt)
            success = True
            
        except Exception as e:
            retries -= 1
            if retries > 0:
                time.sleep(3)
    
    bot_response = r.text
    # print(r.text)

    return bot_response

# print(gensecond_semantic("What is the wavelength used in MgO thin films deposition"))
def chatbot_(user_input):
    r=gensecond_semantic(user_input)
    return r
    # print("Bot said 🤖: ", r)

print(4)