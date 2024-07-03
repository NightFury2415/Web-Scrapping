import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load job data
with open(r'C:\Users\malav\OneDrive\Desktop\job-api\Web-Scrapping\jobs.json') as file:
    jobs = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, jobs):
    if intents_list:
        search_keywords = intents_list[0]['intent'].split()
        matched_jobs = []

        for job in jobs:
            job_description = job['job_title'].lower() + " " + job['company_location'].lower()
            if all(keyword in job_description for keyword in search_keywords):
                matched_jobs.append(job)

        if matched_jobs:
            job = random.choice(matched_jobs)
            result = f"Job Title: {job['job_title']}\nLocation: {job['company_location']}\nCompany: {job['company_name']}\nLink: {job['job_detail_url']}"
        else:
            result = "No matching job found."
    else:
        result = "I didn't understand that. Could you please rephrase?"

    return result

print("ready to roll the Bots")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, jobs)
    print(res)
