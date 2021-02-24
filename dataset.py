import torch 
import json 
import os 
import h5py 
from torch.utils.data import Dataset 
from transformers import GPT2Tokenizer
SPECIAL_TOKENS = ['<bos>', '<eos>', '<img>', '<txt>', '<pad>']
SPECIAL_TOKENS_DICT = {'bos_token':'<bos>', 'eos_token':'eos', 'additional_special_tokens': ['<img>', '<txt>'], 'pad_token':'<pad>'}


class COCODataset(Dataset):

    # output: image region, caption token ids, token_type_ids  
    def __init__(self, data_path, tokenizer):
        self.img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'), 'r')
        train_data_path = os.path.join(data_path, 'annotations')
        with open(os.path.join(train_data_path, 'captions_train2014.json')) as f:
            self.train_data = json.load(f)['annotations'] 
        self.tokenizer = tokenizer 

    def __getitem__(self, i):
        cap_dict = self.train_data[i]
        img_id = str(cap_dict['image_id']) + '_features' 
        img = self.img_features[img_id] 
        # img = torch.FloatTensor(self.img_features[img_id])
        txt = cap_dict['caption'] 
        #txt_ids = torch.Tensor(self.str2id(txt)).long()
        txt_ids = self.str2id(txt)
        token_type_ids, txt_ids = buid_token_type(img, txt_ids, self.tokenizer) 

        img = torch.FloatTensor(img) 
        txt_ids = torch.Tensor(txt_ids).long()
        token_type_ids = torch.Tensor(token_type_ids).long()
        return img, txt_ids, token_type_ids  
    
    def str2id(self, sentence):
        sentence = self.tokenizer.tokenize(sentence) 
        return self.tokenizer.convert_tokens_to_ids(sentence) 
    
    def __len__(self):
        return len(self.train_data) 


def buid_token_type(img, txt_ids, tokenizer): 
    bos, eos, img_id, txt_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    txt_ids = [bos] + txt_ids + [eos] 
    token_type_ids = [img_id] * img.shape[0] + [txt_id] * len(txt_ids)
    # print(token_type_ids)
    return token_type_ids, txt_ids 


if __name__ == "__main__":
    path = 'data' 
    tokenizer = GPT2Tokenizer('model/vocab.json', 'model/merges.txt')
    data = COCODataset(path, tokenizer) 
    img_feature, txt_ids, token_type_ids = data[0]
    print(txt_ids)


