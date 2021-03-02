import numpy as np 
import torch 
import pickle 
from transformers import * 
from dataset import COCODataset 
from VisionGPT import VisionGPT 

SPECIAL_TOKENS_DICT = {'bos_token':'<bos>', 'eos_token':'eos', 'additional_special_tokens': ['<img>', '<txt>'], 'pad_token':'<pad>'}


# create the key-value memory bank 
def memory_bank_construction(model, dataset):
    key_list = []
    value_list = [] 
    mem_bank = {} 
    for instance in dataset:
        img_feature, txt_ids, token_type_ids = instance 
        # enumerate all subsequence 
        img_feature_len = img_feature.size(0)
        for i in range(1, len(txt_ids)-1): 
            t_txt_ids = txt_ids[:i] 
            t_token_type_ids = token_type_ids[:img_feature_len+i] 
            hidden_states = generate_key(img_feature, t_txt_ids, t_token_type_ids, model).tolist() 
            key_list.append(hidden_states)
            value_list.append(txt_ids[i].item())
            #break 
        #break 
    #print(key_list)
    #print(value_list)
    mem_bank['key'] = key_list 
    mem_bank['value'] = value_list 
    return mem_bank 


# generate the history hidden state from model outputs as the key 
def generate_key(img_feature, txt_ids, token_type_ids, model):
    txt_embs = model.transformer.wte(txt_ids) 
    img_embs = model.img_ff(img_feature)
    input_embs = torch.cat((img_embs, txt_embs), 0)
    hidden_state = model(input_embs, token_type_ids)[1][-1]
    return hidden_state 


# store the generated memory bank with pickle 
def mem_bank_store(mem_bank, path):
    f = open(path, 'wb')
    pickle.dump(mem_bank, f)
    f.close()


# employ a trained image captioning model to generate memory bank 
def mem_bank_ini():
    path = 'data' 
    tokenizer = GPT2Tokenizer('model/vocab.json', 'model/merges.txt') 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
    model_config = GPT2Config.from_pretrained('model') 

    model = VisionGPT(model_config)
    model.resize_token_embeddings(len(tokenizer)) 
    model.eval() 
    dataset = COCODataset(path, tokenizer) 
    mem_bank = memory_bank_construction(model, dataset) 
    mem_bank_store(mem_bank, 'data/mem_bank.pkl')


if __name__ == "__main__": 
    mem_bank_ini()
