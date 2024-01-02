from flask import Flask, render_template, request, jsonify
from gensim.models import Word2Vec
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pickle

app = Flask(__name__)

with open('moodmModel.pkl', 'rb') as file:  
        model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    input_text = data.get('input_text', '')

    w2v_data = input_text.split()
    w2v_model = Word2Vec(w2v_data, vector_size=150, window=10, min_count=2, workers=4)

    word_vectors = [w2v_model.wv[word] for word in w2v_data if word in w2v_model.wv.key_to_index]
    res = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(w2v_model.vector_size)

    vectorized_texts = np.array([res])

    text = model.predict(vectorized_texts)

    processed_text = f"You entered: {text}"
    return jsonify({'processed_text': processed_text})

if __name__ == '__main__':
    app.run(debug=True)
