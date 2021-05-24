import glob
import os
from random import shuffle
from nltk.tokenize import TreebankWordTokenizer
from nlpia.loaders import get_data
import yaml
import numpy as np
word_vectors = get_data('w2v', limit=200000)

classes = set()

def pre_process_data(filepath):
    with open(filepath) as file:
        documents = yaml.full_load(file)
        intents = documents['nlu']
        dataset = []
        for intent in intents:
            intent_name = intent['intent']
            classes.add(intent_name)
            for example in intent['examples']:
                dataset.append((intent_name, example))
        shuffle(dataset)
        return dataset

dataset = pre_process_data('mood.yml')
classes = list(classes)


def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    expected = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass
        vectorized_data.append(sample_vecs)
    return vectorized_data


def collect_expected(dataset):
    expected = []
    for sample in dataset:
        one_hot = np.zeros(len(classes))
        one_hot[classes.index(sample[0])] = 1
        expected.append(one_hot)
    return expected



vectorized_data = tokenize_and_vectorize(dataset)

expected = collect_expected(dataset)

split_point = int(len(vectorized_data) * .8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]


maxlen = 400
batch_size = 32
embedding_dims = 300
epochs = 100

def pad_trunc(data, maxlen):
    new_data = []
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)



x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Activation
num_neurons = 50
model = Sequential()

model.add(LSTM(
  num_neurons, return_sequences=True,
  input_shape=(maxlen, embedding_dims)))

model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(len(classes)))
model.add(Activation('softmax'))

model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))




# ==============
vec_list = tokenize_and_vectorize([('mood_baaah', 'I am not feeling too well')])
test_vec_list = pad_trunc(vec_list, maxlen)

test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
model.predict(test_vec)
classes[np.argmax(model.predict(test_vec))]