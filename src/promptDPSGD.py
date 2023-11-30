
"""

"""
import torch 

class PromptDPSGD():
    def __init__(self, method, epsilon, llm, training_iteration, 
                 max_gradient_norm, sequence_length, embedding_dim=512,
                 ):
        self.method = method
        #TODO if method
        self.epsilon = epsilon
        self.llm = llm
        self.training_iteration = training_iteration
        self.max_gradient_norm = max_gradient_norm
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        
    def prepend (self,):
        pass
    
    def promptDPSGD (self):
        pt = torch.rand((self.sequence_length, self.embedding_dim))
        for t in range(self.training_iteration):
            
        
        