import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load jobs data from JSON file
with open(r'C:\Users\malav\OneDrive\Desktop\job-api\Web-Scrapping\jobs.json') as file:
    jobs = json.load(file)

# Load pickled data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
    return res[0] > 0.5

def get_response(user_input, jobs):
    user_keywords = clean_up_sentence(user_input)
    matched_jobs = []

    for job in jobs:
        job_description = (job['job_title'] + " " + job['company_location']).lower()
        if any(keyword in job_description for keyword in user_keywords):
            matched_jobs.append(job)

    # Randomly choose 5 jobs if there are matches
    if matched_jobs:
        # Ensure we don't exceed the number of available jobs
        num_results = min(5, len(matched_jobs))
        random_jobs = random.sample(matched_jobs, num_results)
        results = []

        for job in random_jobs:
            result = f"Job Title: {job['job_title']}\nLocation: {job['company_location']}\nCompany: {job['company_name']}\nLink: {job['job_detail_url']}\n"
            results.append(result)
        
        return results
    else:
        return ["No matching jobs found."]

print("Ready to roll the Bots")

while True:
    message = input("")
    is_job = predict_class(message)
    if is_job:
        res = get_response(message, jobs)
        for result in res:
            print(result)
    else:
        print("Sorry, I didn't understand that. Please try again.")
