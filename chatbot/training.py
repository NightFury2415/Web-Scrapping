import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

with open(r'C:\Users\malav\OneDrive\Desktop\job-api\Web-Scrapping\jobs.json') as file:
    jobs = json.load(file)

words = []
documents = []
ignore_letters = ['?', '/', '.', ',', '!']
classes = ['job']

for job in jobs:
    combined_text = job['job_title'] + " " + job['company_location']
    word_list = nltk.word_tokenize(combined_text)
    words.extend(word_list)
    documents.append((word_list, 'job'))

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
for document in documents:
    bag = []
    words_patterns = document[0]
    words_patterns = [lemmatizer.lemmatize(word.lower()) for word in words_patterns]
    bag = [1 if word in words_patterns else 0 for word in words]
    training.append([bag, [1]])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array([np.array(item[0]) for item in training])
train_y = np.array([np.array(item[1]) for item in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=200, batch_size=5)

model.save('chatbot_model.h5', hist)
print("Done!!!")
