
"""

"""
import torch 
import random
from torch.nn import CrossEntropyLoss


class PromptDPSGD():
    def __init__(self, method, epsilon, llm, sampling_rate, training_iteration, 
                 max_gradient_norm, noise_scale, sequence_length, embedding_dim, 
                 lr
                 ):
        self.method = method
        #TODO if method
        self.epsilon = epsilon
        self.llm = llm
        self.sampling_rate = sampling_rate
        self.training_iteration = training_iteration
        self.max_gradient_norm = max_gradient_norm
        self.noise_scale = noise_scale
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.loss_fn = CrossEntropyLoss()
        
    def _llm_inner_embedding_output (self, xi, prompt_embedding):
        return self.llm.bert.embeddings(xi) + prompt_embedding
    
    def _llm_withou_inner_embedding (self, embeddings_output):
        yi = self.llm.bert.encoder(embeddings_output)
        yi = self.llm.bert.pooler(yi[0])
        yi = self.llm.dropout(yi)
        return self.llm.classifier(yi)
    
    def prompt_learning (self, batch):
        pt, privacy_cost = self._promptDPSGD(batch)
        
        
    def _promptDPSGD (self, batch):
        pt = torch.rand((1, self.sequence_length, self.embedding_dim), requires_grad=True)
        for t in range(self.training_iteration):
            Bt_index = random.sample(range(0, len(batch[list(batch.keys())[0]])), 
                               self.sampling_rate*len(batch[list(batch.keys())[0]]))
            gts=[]
            for i in Bt_index:
                lp_output = self._llm_withou_inner_embedding(
                    self._llm_inner_embedding_output(batch['input_ids'][i].unsqueeze(0), 
                                                     pt))[0].detach()
                lp_output.requires_grad = True
                loss = self.loss_fn(lp_output, batch['label'][i].unsqueeze(0))
                gt_xi = torch.autograd.grad(loss, pt, allow_unused=True)[0]
                gt_xi_clip = gt_xi/max(1, torch.norm(gt_xi, p=2)/self.max_gradient_norm)
                gts.append(gt_xi_clip)
            gt_hat = (1/len(Bt_index))*(sum(gts)+torch.normal(
                0, self.noise_scale**2*self.max_gradient_norm**2*torch.eye(
                    self.sequence_length, self.embedding_dim)).unsqueeze(0))
            pt=pt-(self.lr*gt_hat)
        
        return pt, self.privacy_cost()
    
    def privacy_cost (self, ):
        pass
            
                
    def _update_model (self):
        pass
            
        
        