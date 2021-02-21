import torch 
import json 
import os 
import h5py 
from torch.utils.data import Dataset 
from transformers import BartTokenizer


class COCODataset(Dataset):

    # output: image region, caption token ids 
    def __init__(self, data_path, tokenizer):
        self.img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'), 'r')
        train_data_path = os.path.join(data_path, 'annotations')
        with open(os.path.join(train_data_path, 'captions_train2014.json')) as f:
            self.train_data = json.load(f)['annotations'] 
        self.tokenizer = tokenizer 

    def __getitem__(self, i):
        cap_dict = self.train_data[i]
        img_id = str(cap_dict['image_id']) + '_features' 
        img = torch.FloatTensor(self.img_features[img_id])
        txt = cap_dict['caption'] 
        txt_ids = torch.Tensor(self.str2id(txt)).long()
        return img, txt_ids 
    
    def str2id(self, sentence):
        sentence = self.tokenizer.tokenize(sentence) 
        return self.tokenizer.convert_tokens_to_ids(sentence) 
    
    def __len__(self):
        return len(self.train_data) 


if __name__ == "__main__":
    path = 'data' 
    tokenizer = BartTokenizer('model/vocab.json', 'model/merges.txt')
    data = COCODataset(path, tokenizer) 
    img_feature, txt_ids = data[0]
    print(txt_ids)



