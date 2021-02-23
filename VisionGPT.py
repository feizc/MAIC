from transformers import * 
import torch.nn as nn 
import torch 
import os 

class Captioner(TransfoXLPreTrainedModel): 
    def __init__(self, config):
        self.transformer = TransfoXLModel(config) 
        

