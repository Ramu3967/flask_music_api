import argparse
import os
import json

import numpy as np
import pickle

from model import build_model, load_weights

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
from collections import Counter
from scipy.stats import entropy

DATA_DIR = './data'
MODEL_DIR = './model'

def build_sample_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))
    for i in range(3):
        model.add(LSTM(256, return_sequences=(i != 2), stateful=True))
        model.add(Dropout(0.2))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    return model

def sample(epoch, header, num_chars):
    with open(os.path.join(DATA_DIR, 'char_to_idx_1.json')) as f:
        char_to_idx = json.load(f)
    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)
    print('vocab_size:',vocab_size)
    model = build_sample_model(vocab_size)
    load_weights(epoch, model)
    model.save(os.path.join(MODEL_DIR, 'model.{}.h5'.format(epoch)))


    sampled = [char_to_idx[c] for c in header]
    print(sampled)
    

    for i in range(num_chars):
        batch = np.zeros((1, 1))
        if sampled:
            batch[0, 0] = sampled[-1]
        else:
            batch[0, 0] = np.random.randint(vocab_size)
        result = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(vocab_size), p=result)
        sampled.append(sample)

    res=''.join(idx_to_char[c] for c in sampled)
    k = res.find("X:")
    if k != -1:
        res = res[k:]
        ind = res.find("\n\n\n")
        return res[:ind + 1]
    return ""



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Sample some text from the trained model.')
    # parser.add_argument('epoch', type=int, help='epoch checkpoint to sample from')
    # parser.add_argument('--seed', default='', help='initial seed for the generated text')
    # parser.add_argument('--len', type=int, default=2048, help='number of characters to sample (default 512)')
    # args = parser.parse_args()

    # print(sample(args.epoch, args.seed, args.len))
    # print(sample(epoch=100, header='', num_chars=1024))

    vocab_size,epoch=87,100

    model = build_sample_model(vocab_size)
    load_weights(epoch, model)
    model.save(os.path.join(MODEL_DIR, 'model.{}.h5'.format(epoch)))

    # with open(os.path.join(DATA_DIR, 'char_to_idx_1.json')) as f:
    #     char_to_idx = json.load(f)
    # idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    # data=[model,char_to_idx,idx_to_char]

    pickle.dump(model,open('model.pkl','wb'))

#     # original ABC music data
#     original_abc = """X: 119
# T:Haymaker's Jig
# % Nottingham Music Database
# S:Trad, arr Phil Rowe
# M:6/8
# K:G
# B/2-c/2|"G"d2d dBd|gfg B3|"Am"cBc ABc|"D7"e2d B2c|
# "G"d2d dBd|"G"gfg B2g|"D"fed ecA|"G"G3 G2::
# B/2-d/2|"G"g2g gfe|dgd B3|"Am"cBc ABc|"D7"e2d B2d|
# "G"g2g gfe|dgd B2g|"D7"fed ecA|"G"G3 -G2:|"""
#
#     # generated ABC music data
#     generated_abc = """X: 119
# T:Haymaker's Jig
# % Nottingham Music Database
# S:Trad, arr Phil Rowe
# M:6/8
# K:G
# "G"d2d dCd|gfg B3|B/2-c/2|"Am"cBc ABc|"D7"e2d B2c|
# "D7"e2d B2d|"D"abb B2g|"D"fed ecA|"B"F3 C2::"G"c3 f2::"A"G3 b2::
# B/2-d/2|"E"f2a gfe|cdg B3|"Am"cBc ABc|"G"d2d dBd|
# "G"g2g gfe|dgd B2g|"D7"fed ecA|"G"G3 -G2:|"""
#
#     # convert ABC notation to sequences of notes represented as integers
#     original_notes = abc_to_notes(original_abc)
#     generated_notes = abc_to_notes(generated_abc)
#
#     # calculate probability distribution of notes in original and generated music
#     original_note_probs = calculate_note_probabilities(original_notes)
#     generated_note_probs = calculate_note_probabilities(generated_notes)
#
#     # calculate KL divergence between probability distributions
#     kl_divergence = calculate_kl_divergence(original_note_probs, generated_note_probs)
#     print("KL Divergence: ", kl_divergence)
