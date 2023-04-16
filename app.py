import pickle
import numpy as np

from flask import Flask,jsonify,render_template
from sample import build_sample_model
from model import load_weights

app = Flask(__name__)

# model, char_to_idx, idx_to_char = load_weights(epoch, model)
char_to_idx={"\n": 0, " ": 1, "!": 2, "\"": 3, "#": 4, "%": 5, "&": 6, "'": 7, "(": 8, ")": 9, "+": 10, ",": 11, "-": 12, ".": 13, "/": 14, "0": 15, "1": 16, "2": 17, "3": 18, "4": 19, "5": 20, "6": 21, "7": 22, "8": 23, "9": 24, ":": 25, "=": 26, ">": 27, "?": 28, "A": 29, "B": 30, "C": 31, "D": 32, "E": 33, "F": 34, "G": 35, "H": 36, "I": 37, "J": 38, "K": 39, "L": 40, "M": 41, "N": 42, "O": 43, "P": 44, "Q": 45, "R": 46, "S": 47, "T": 48, "U": 49, "V": 50, "W": 51, "X": 52, "Y": 53, "[": 54, "\\": 55, "]": 56, "^": 57, "_": 58, "a": 59, "b": 60, "c": 61, "d": 62, "e": 63, "f": 64, "g": 65, "h": 66, "i": 67, "j": 68, "k": 69, "l": 70, "m": 71, "n": 72, "o": 73, "p": 74, "q": 75, "r": 76, "s": 77, "t": 78, "u": 79, "v": 80, "w": 81, "x": 82, "y": 83, "z": 84, "|": 85, "~": 86}
idx_to_char= { i: ch for (ch, i) in char_to_idx.items() }

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/generate_music',methods=['GET'])
def generate_music():
    vocab_size, epoch,num_chars = 87, 100,1024
    model = build_sample_model(vocab_size)
    load_weights(epoch, model)
    sampled = [char_to_idx[c] for c in '']
    for i in range(num_chars):
        batch = np.zeros((1, 1))
        if sampled:
            batch[0, 0] = sampled[-1]
        else:
            batch[0, 0] = np.random.randint(vocab_size)
        result = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(vocab_size), p=result)
        sampled.append(sample)

    res = ''.join(idx_to_char[c] for c in sampled)
    k = res.find("X:")
    if k != -1:
        res = res[k:]
        ind = res.find("\n\n\n")
        original_music_string=res[:ind + 1]

        # modified_string=modified_string.replace('\n','')
        # modified_string=modified_string.replace('|', '|\n')
        # return jsonify({'music_sequence': original_music_string}), 200, {'Content-Type': 'application/json; charset=utf-8',
        #                                                     'Content-Encoding': 'identity', 'Cache-Control': 'no-cache'}
        # return jsonify({"music_sequence": modified_string})
        # modified_string=original_music_string.replace("\n", "<br>")

        print(original_music_string)
        data={"music_sequence": original_music_string}
        response=jsonify(data)
        response.headers.add('Content-Type', 'application/json')
        return response


    return jsonify({"music_sequence": ''})



if __name__ == '__main__':
    app.run(debug=True)
