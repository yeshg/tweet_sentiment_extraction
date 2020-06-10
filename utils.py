import pandas as pd

import numpy as np
import pickle
import bcolz

import torch

EMB_DIM = 50 # 50 if using glove6B/6B.50.dat

def get_selected_text(text, start_idx, end_idx):
    words = text.split()
    selected_text = ""
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, len(words)-1)
    for ix in range(start_idx, end_idx+1):
        selected_text += words[ix]
        if ix < end_idx:
            selected_text += " "
    return selected_text

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred)
        
    true = get_selected_text(text, start_idx, end_idx)
    
    return jaccard(true, pred)

def load_data():

    train_data = pd.read_csv("data/train.csv", skiprows=0, header=0)
    test_data = pd.read_csv("data/test.csv", skiprows=0, header=0)
    # print(train_data.head())
    # print("Train data shape: {}".format(train_data.shape))
    # print(test_data.head())
    # print("Test data shape: {}".format(test_data.shape))
    train_data.dropna(how='any', inplace=True)
    test_data.dropna(how='any', inplace=True)
    return train_data, test_data

def process_train_data(data, word_to_ix):

    def find_range(str1, str2):
        start_ind = str1.find(str2)
        str1_words = str1.split()
        prev_count = 0
        count = 0
        for i, e in enumerate(str1_words):
            if start_ind >= count and start_ind <= count+len(e):
                return i, i+len(str2.split())-1
            count += len(e)
        return 0, len(str2.split())-1

    # Find start and end indices. this is the output
    outputs = np.zeros((data.shape[0]+1, 2))
    for index, row in data.iterrows():
        a, b = find_range(row['text'], row['selected_text'])
        outputs[index, 0] = a
        outputs[index, 1] = b
    Y_all = torch.tensor(outputs, dtype=torch.long)

    inputs = []
    for index, row in data.iterrows():
        sentiment = torch.tensor(word_to_ix[row['sentiment']], dtype=torch.long).reshape(1)
        text_tensors = torch.tensor([word_to_ix[w] for w in row['text'].lower().split(" ")], dtype=torch.long)
        inputs.append(torch.cat([sentiment, text_tensors]))
    X_all = inputs

    # Split data into train, val and test
    val_num = 500
    # indices = np.random.permutation(Y_all.shape[0])
    # train_inds, val_inds = indices[val_num:], indices[:val_num]
    X_train, X_val = X_all[val_num:], Y_all[:val_num]
    Y_train, Y_val = Y_all[val_num:], Y_all[:val_num]

    return X_train, Y_train, X_val, Y_val

# def process_test_data(data):

#     def find_range(str1, str2):
#         start_ind = str1.find(str2)
#         str1_words = str1.split()
#         prev_count = 0
#         count = 0
#         for i, e in enumerate(str1_words):
#             if start_ind >= count and start_ind <= count+len(e):
#                 return i, i+len(str2.split())-1
#             count += len(e)
        
#     # Find start and end indices. this is the output
#     outputs = data.apply(lambda row : find_range(row['text'], row['selected_text']), axis=1)

#     # stack inputs together for ease of putting into model
#     inputs = data[['text', 'sentiment']]
#     inputs = inputs['sentiment'].str.cat(inputs['text'],sep=" ")

#     print(inputs.iloc[20])
#     print()
#     print(outputs.iloc[20])

#     return inputs, outputs

# Input is dataset and glove mapping
def get_embedding(dataset, glove):

    target_vocab = set(["neutral", "positive", "negative"])
    dataset.text.apply(lambda x: target_vocab.update((str(x).lower().split(" "))))

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0

    for i, word in enumerate(target_vocab):
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(EMB_DIM, ))

    word_to_ix = {word: i for i, word in enumerate(target_vocab)}

    return torch.Tensor(weights_matrix), word_to_ix
# def create_dictionary_mapping():


# original source: https://medium.com/@martinpella/
def process_glove_vectors():

    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='glove6B/glove.6B.50.dat', mode='w')

    with open('glove6B/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    # actually are 400K + 1 <unk> words in glove
    vectors = bcolz.carray(vectors[1:].reshape((-1, 50)), rootdir='glove6B/glove.6B.50.dat', mode='w')
    vectors.flush()
    with open('glove6B/glove.6B.50_words.pkl', 'wb') as f:
        pickle.dump(words, f)
    with open('glove6B/glove.6B.50_idx.pkl', 'wb') as f:
        pickle.dump(word2idx, f)

def load_glove_vectors():

    vectors = bcolz.open('glove6B/glove.6B.50.dat')[:]
    words = pickle.load(open('glove6B/glove.6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open('glove6B/glove.6B.50_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}

    # print(glove['the'])

    return glove

