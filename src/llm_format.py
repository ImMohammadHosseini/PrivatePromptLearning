
"""

"""
import torch
from pytorch import nn
from transformers import AutoModelForSequenceClassification


class LLM_Format (nn.Module):
    """
    """

    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForSequenceClassification.from_pretrained("prajjwal1/"+ model_name)
    
    def get_encoder_layer_num (self):
        return len(self.llm.bert.encoder.layer)
    
    def llm_inner_embedding_output (self, xi, prompt_embedding):
        return self.llm.bert.embeddings(xi) + prompt_embedding
    
    def llm_withou_inner_embedding (self, embeddings_output):
        yi = self.llm.bert.encoder(embeddings_output)
        yi = self.llm.bert.pooler(yi[0])
        yi = self.llm.dropout(yi)
        return self.llm.classifier(yi)
    
    def forward(self, xis, pts_list = None):
        if pts_list == None:
            return self.llm(xis['input_ids'])
        
        elif len(pts_list) == 1:
            return self.llm_withou_inner_embedding(
                self.llm_inner_embedding_output(xis['input_ids'], 
                                                torch.cat(pts_list*xis['input_ids'].size(0), 0)))[0]#.detach()
        elif len(pts_list) > 1:
            pass