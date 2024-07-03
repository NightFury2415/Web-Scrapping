import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

lemmatizer = WordNetLemmatizer()

# Load jobs data
with open(r'C:\Users\malav\OneDrive\Desktop\job-api\Web-Scrapping\jobs.json') as file:
    jobs = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '/', '.', ',', '!']

# Process jobs data
for job in jobs:
    job_title = job.get('job_title')
    job_location = job.get('company_location')
    
    if job_title and job_location:
        combined_text = job_title + " " + job_location
        word_list = nltk.word_tokenize(combined_text)
        words.extend(word_list)
        documents.append((word_list, 'job'))
        if 'job' not in classes:
            classes.append('job')

# Debug: Print number of jobs processed
print(f"Total jobs processed: {len(documents)}")

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

with open('words.pkl', 'wb') as file:
    pickle.dump(words, file)
with open('classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    words_patterns = document[0]
    words_patterns = [lemmatizer.lemmatize(word.lower()) for word in words_patterns]
    bag = [1 if word in words_patterns else 0 for word in words]

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array([np.array(item[0]) for item in training])
train_y = np.array([np.array(item[1]) for item in training])

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Use Adam optimizer with a custom learning rate
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Implement early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)

hist = model.fit(train_x, train_y, epochs=300, batch_size=8, callbacks=[early_stopping, reduce_lr])

model.save('chatbot_model.h5')
print("Done!!!")
