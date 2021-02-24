import numpy as np 
from transformers import * 
from dataset import COCODataset 
from VisionGPT import VisionGPT 

SPECIAL_TOKENS_DICT = {'bos_token':'<bos>', 'eos_token':'eos', 'additional_special_tokens': ['<img>', '<txt>'], 'pad_token':'<pad>'}


def memory_bank_construction(model, dataset):
    return 1




if __name__ == "__main__": 
    
    path = 'data' 
    tokenizer = GPT2Tokenizer('model/vocab.json', 'model/merges.txt') 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
    model_config = GPT2Config.from_pretrained('model') 

    model = VisionGPT(model_config) 
    model.eval() 
    dataset = COCODataset(path, tokenizer) 
    
