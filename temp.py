# =================== Imports ===================== #
import torch
from joblib import dump, load
from model import InferSent

print("Imported libraries")

# ========================= Setup model ============================ #
model_version = 1
MODEL_PATH = "./weights.pkl"
W2V_PATH = './word_dict.txt'
VOCAB_SIZE = 50000  

params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
print("Initialised model")

model.load_state_dict(torch.load(MODEL_PATH))
model.set_w2v_path(W2V_PATH)
print("Now building vocabulary")
model.build_vocab_k_words(K=VOCAB_SIZE)

# ======================== Save Model ========================= #
print("Now saving")
dump(model, './model4.joblib', compress=4)


'''
# ================= Messing with Code ================================= #
import torch
from collections import OrderedDict
from data.model import InferSent
print("Imported libraries")

w2v = load("./data/w2v9.joblib")
weights = load("./data/weights9.joblib")
# Print model's state_dict
reg_dict = OrderedDict()
rev_dict = OrderedDict()
for param_tensor in weights:
    if(param_tensor[-8:] == "_reverse"): 
        rev_dict[param_tensor] = weights[param_tensor]
    else:
        reg_dict[param_tensor] = weights[param_tensor]
dump(reg_dict, "./weights9_reg.joblib", compress=9)
dump(rev_dict, "./weights9_rev.joblib", compress=9)
'''