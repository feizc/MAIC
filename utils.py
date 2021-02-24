import numpy as np 
import torch 
from transformers import * 
from dataset import COCODataset 
from VisionGPT import VisionGPT 

SPECIAL_TOKENS_DICT = {'bos_token':'<bos>', 'eos_token':'eos', 'additional_special_tokens': ['<img>', '<txt>'], 'pad_token':'<pad>'}


def memory_bank_construction(model, dataset):
    for instance in dataset:
        img_feature, txt_ids, token_type_ids = instance 
        generate_key(img_feature, txt_ids, token_type_ids, model)
        break 


def generate_key(img_feature, txt_ids, token_type_ids, model):
    txt_embs = model.transformer.wte(txt_ids) 
    img_embs = model.img_ff(img_feature)
    input_embs = torch.cat((img_embs, txt_embs), 0)
    res = model(input_embs, token_type_ids)
    print(res[1][-1].size())


if __name__ == "__main__": 
    
    path = 'data' 
    tokenizer = GPT2Tokenizer('model/vocab.json', 'model/merges.txt') 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
    model_config = GPT2Config.from_pretrained('model') 

    model = VisionGPT(model_config)
    model.resize_token_embeddings(len(tokenizer)) 
    model.eval() 
    dataset = COCODataset(path, tokenizer) 
    memory_bank_construction(model, dataset) 

