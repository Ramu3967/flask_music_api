import pickle
import numpy as np

from flask import Flask,jsonify,render_template

app = Flask(__name__)

model, char_to_idx, idx_to_char = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/generate_music',methods=['GET'])
def generate_music():
    sampled = [char_to_idx[c] for c in '']
    vocab_size, epoch,num_chars = 87, 100,1024
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
        modified_string=original_music_string.replace("\n", "<br>")

        print(original_music_string,modified_string,sep="\n\n\n")
        data={"music_sequence": modified_string}
        response=jsonify(data)
        response.headers.add('Content-Type', 'application/json')
        return response


    return jsonify({"music_sequence": ''})



if __name__ == '__main__':
    app.run(debug=True)
