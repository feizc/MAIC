from transformers import * 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 
import torch 
import os 


class VisionGPT(GPT2PreTrainedModel): 

    def __init__(self, config):
        super(VisionGPT, self).__init__(config) 
        self.transformer = GPT2Model(config) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
        self.img_ff = nn.Linear(2048, config.n_embd) 

        self.init_weights()
        self.tie_weights() 
    
    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte)

    def forward(self, input_embs, token_type_ids, labels=None):
        transformer_outputs = self.transformer(inputs_embeds=input_embs, token_type_ids=token_type_ids) 
        hidden_states = transformer_outputs[0] 

        lm_logits = self.lm_head(hidden_states) 
        outputs = (lm_logits,) + transformer_outputs
        if labels is not None: 
            loss_text_fct = CrossEntropyLoss(ignore_index=-100) 
            loss_text = loss_text_fct(lm_logits.squeeze(0), labels.squueeze(0)) 
            outputs = (loss_text, ) + outputs 
        return outputs 






