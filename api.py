print("Importing and initialising")
# ========================== Imports =================================== #
from flask import Flask
from flask_restful import Resource, Api, reqparse
from joblib import load
from model import InferSent
import numpy as np

# ========================== Initialisation =================================== #
app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('entries', required=True, action='append', help="Need at least two text entries")

w2v = load("./data/w2v9.joblib")
weights_reg = load("./data/weights9_reg.joblib")
weights_rev = load("./data/weights9_rev.joblib")
tokenizer = load("./data/english.pickle")


# ===================== Build Model ============================== #
print("building model")
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0}
model = InferSent(params_model)
weights_reg.update(weights_rev)
model.load_state_dict(weights_reg)
model.set_w2v(w2v)


# ========================== HELPER FUNCTIONS ================================ #
print("Helper functions and routes")
def similarity(vec1, vec2):
    numerator = np.dot(vec1, vec2)
    denominator = np.sqrt(np.sum(vec1 ** 2)) * np.sqrt(np.sum(vec2 ** 2))
    return numerator / denominator

def doc_similarity(entries):
    sentences = []; embeddings = []
    for entry in entries:
        sentences.append(load_sentence(entry))
        embeddings.append( model.encode(sentences[-1], bsize=128, tokenize=False, verbose=True) )
    
    avg_similarity = 0
    num_sentences = 0
    #Now compute similarity of each sentence with every other sentence
    for sent1 in embeddings[0]:
        for sent2 in embeddings[1]:
            sim = similarity(sent1, sent2)
            avg_similarity += sim
            num_sentences += 1
            
    return avg_similarity / num_sentences

def load_sentence(text):
    punctuation = '!"#$%&'+"()*+, -./:;<=>?@[\]^_`{|}~"
    #Creates a dict where each punctuation symbol maps to that symbol with spaces around it.
    mapping = {key: f" {key} " for key in punctuation}
    #str.translate replaces each key in dict with val in dict. Except special representation of dict that str.maketrans returns
    padded = text.translate(str.maketrans(mapping)) 
    separ = tokenizer.tokenize(padded)
    return separ



# ========================= Setup routes ===================== #
class compare(Resource):    
    def post(self):
        args = parser.parse_args()
        entries = args['entries']  
        return doc_similarity(entries)
        
api.add_resource(compare, '/')

if __name__ == '__main__':
    app.run(debug=True)