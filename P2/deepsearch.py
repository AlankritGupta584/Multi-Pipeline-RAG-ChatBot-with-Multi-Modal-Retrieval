# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import requests
import json
import random
import time
import numpy as np
import re
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


def semantic_search2(query, k=200):
    ans=[]
    query_e=generate_embeddings(query)
    results = index.query(
        vector=query_e.tolist(),
        top_k=k,
        include_metadata=True
    )
    for match in results.matches:
        ans.append(match["metadata"]["text"])
    return ans



def material_finder(query):
    Prompt=f"""
    Your task is to find all materials or substrates mentioned in the following text. If no substrate present, type "None" in inverted commas.
    For each material: list all possible names, synonyms, abbreviations, and alternative representations. Don't mix it up with other compounds.

    1. Return each group as a comma-separated list, enclosed in inverted commas ("").
    2. Include chemical formulas, common names, IUPAC names, historical name, informal names, research paper names and industry abbreviations.
    3. While writing formula, don't write numbers in subscript or superscript or with underscore. for example: O2 not O_2

    Text:
    {query}

    Example input:
    "Give me the wavelength for LSMO thin films deposition over Al2O3 substrate using KrF excimer laser."

    Expected output:
    "LSMO","Lanthanum Strontium Manganite", "Lanthanum Strontium Manganese Oxide", "LaSrMnO3", "LaSrMO", "La1-xSrxMnO3", "Perovskite Manganite",
    "Al2O3", "Al(III) oxide", "Aluminium oxide", "Aluminium(III) oxide", "AAO", "alumina", "alundum", "aloxide", "Alpha-Alumina",
    "KrF", "Krypton fluoride"


    """

    retries = 3 
    success = False
    
    while retries > 0 and not success:
        try:
            model = get_random_model()
            response = model.generate_content(Prompt)
            success = True
            
        except Exception as e:
            retries -= 1
            if retries > 0:
                time.sleep(3)
    
    response=response.text
    # response = ollama(Prompt)

    result = re.findall(r'"(.*?)"', response)
    return result



##################################################
def parameter_finder(query):

    


    Prompt=f"""
    Your task it to find all parameters/specifications like wavelength, pressure, temperature, laser fluence, etc in the given text. 
    Return each as a comma-separated list, enclosed in inverted commas ("").
    Explore as many variants it can take in other text.
    If pressure is for example 1mPa, make sure to also include 1×10-3 Pa, 10-3 Pa, 1×10-3Pa, 10-3Pa. Don't write any number in super or subscript

    Example input:
    "Is 300 C temperature good when I use 248 nm and 200 pulsed laser on ZnO?"

    Expected output:
    "300", "300°C", "300 °C", "300 celsius, "248 nm", "248nm", "248 nanometer", ""248nanometer", "248", "200 pulse", "200 pulsed".

    Text:
    {query}

    """
    retries = 3 
    success = False
    
    while retries > 0 and not success:
        try:
            model = get_random_model()
            response = model.generate_content(Prompt)
            success = True
            
        except Exception as e:
            retries -= 1
            if retries > 0:
                time.sleep(3)
    

    result = re.findall(r'"(.*?)"', response.text)
    return result




def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def embedding_map_generator(query):
    filtered_para=filter(query)

    embedding_map = [] 

    print("Initiating Semantic Search 💿... ")
    print("...")
    for para in filtered_para:
        embedding = np.array(generate_embeddings(para))
        embedding_map.append((para, embedding))
    
    return embedding_map

def semantic_search(query, top_k=15):
    query_vec = np.array(generate_embeddings(query))

    embedding_map=embedding_map_generator(query)
    # print(len(embedding_map))

    print("Please type k, try to keep around 10 or 15. More k means more accuracy, but slow confused result.\nLess k means less accuracy, but faster.")
    top_k=int(input("Number of top k paragraphs you want (out of 20): "))
    scored = []
    for para, vec in embedding_map:
        score = cosine_similarity(query_vec, vec)
        scored.append((para, score))
    
    top_paragraphs = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    print("Semantic Search Done 🧠...")
    print("...")
    return [para for para, _ in top_paragraphs]

def filter(query):
    paragraphs=semantic_search2(query)
    materials=material_finder(query)
    parameters=parameter_finder(query)
    from collections import defaultdict

    unique_paragraphs = list(set(paragraphs))

    count = defaultdict(int)

    print("Finding relevant paragraphs 📃...")
    
    for para in unique_paragraphs:
        para_lower = para.lower()
        matched = set()
        for material in materials:
            if material.lower() in para_lower:
                matched.add(material)
        for param in parameters:
            if param.lower() in para_lower:
                matched.add(param)
        count[para] = len(matched)

    sorted_paras = sorted(count.items(), key=lambda x: x[1], reverse=True)

    filtered_para = [para for para, _ in sorted_paras[:50]]
    return filtered_para


def summarizer(query):
    sim = semantic_search(query)
    print("Analysing the retrieved paragraphs 🕵🏻🔎...")
    print("...")
    
    summ = []
    for para in sim:
        retries = 3 
        success = False
        
        while retries > 0 and not success:
            try:
                model = get_random_model()
                response = model.generate_content(f"""
                                                  Summarize the given text in two lines.
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


def genfirst(query, summ):
    prompt = f"""
    You are given query. Find the most appropriate one or multiple paragraphs.
    Also state reason and mention every important information related to query.
    
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


import random

waiting_messages = [
    "We apologise for the delay, our bot was having a nap.",
    "We apologize for the delay—our bot was dreaming of electric sheep.",
    "Hold tight! We’re just bribing the servers with coffee.",
    "Please wait… Our tech wizard is untangling some very complicated wires (and emotions).",
    "Things are moving slower than a turtle on vacation. Thanks for your patience!",
    "Hang on! We’re trying to contact HQ... but the aliens aren’t picking up.",
    "Loading… like your uncle’s dial-up internet.",
    "Just a moment! Our code monkey is putting on its thinking top hat.",
    "We’re working on it! Somewhere, a hamster is sprinting frantically in a wheel.",
    "Delay detected: Brain.exe is updating. Please wait.",
    "Hold on—we're tracking down the problem. It might be hiding in the vents.",
    "Our servers are taking a pizza break. They’ll be back soon, fully fueled!",
    "Slow as a sloth on Sunday… but we’re getting there!",
    "Unicorns are delivering your request, but they’re a bit magical and slow today.",
    "Your data is packed in a box… and the delivery guy took a detour.",
    "Our wizard is casting the ‘load’ spell… please hold your applause.",
    "Bot reboot in progress. It’s thinking deep thoughts about existence.",
    "Hang on, you’re on the loading rollercoaster—thrills and chills ahead!",
    "Patience, young grasshopper. The snails are doing their part.",
    "Party’s starting soon! Just waiting for the last guest (your data) to arrive.",
    "Putting together your request like a 10,000-piece puzzle… almost done!"
]




def querygen(query):

    prompt=f"""
    You are given a query. Your task to short this query as much as you can. You should include material/substrate if mention any. Parameter if mentioned any. Specification if mentioned any.
    Return the answer in inverted comma.

    For example:
    if query is "What is the wavelength of the ZnO films on Al2O3 substrate?", then return "Wavelength of ZnO on Al2O3"
    if query is "Can I use 248 nm wavelength of KrF excimer to prepare Sb2Te3?" then return "248 nm KrF on Sb2Te3"
    if query is "What temperature to maintain in PLD?" then return "temperature in PLD"

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
            retries -= 1
            if retries > 0:
                time.sleep(3)

    r=r.text

    return r

conversation_history = []

def gensecond_deep(query):
    print("Working over your query✅...")
    print("...")

    
    summ = summarizer(query)

    random_message = random.choice(waiting_messages)
    print(random_message)


    r1 = genfirst(query, summ)

    print("Generating your answer✍🏻...")



    prompt = f"""
You have a query and the hint to the query about PLD (Pulsed Laser Deposition).

1. Present to me it like you didn't get any hint and you're speaking it using your data. DON'T MENTION ANYTHING ABOUT PARAGRAPHS EXPLICITLY.
2. Present every information. Dont miss anything, you have to make it look like very deep analysis.

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

    return bot_response



print(gensecond_deep("What is the wavelength used in MgO thin films deposition"))
def chatbot_(user_input):
    r=gensecond_deep(user_input)
    return r
    # print("Bot said 🤖: ", r)

print(4)