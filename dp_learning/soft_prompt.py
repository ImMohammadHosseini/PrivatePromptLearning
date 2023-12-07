
"""

"""
import torch 
import random
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from opacus import PrivacyEngine

from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm.notebook import tqdm
import numpy as np


class Soft_Prompt():
    def __init__(self, llm, epsilon,delta, sampling_rate, training_iteration, 
                 max_gradient_norm, noise_scale, sequence_length, embedding_dim, 
                 lr, device
                 ):
        
        self.llm = llm
        self.epsilon = epsilon
        self.delta = delta
        self.sampling_rate = sampling_rate
        self.training_iteration = training_iteration
        self.max_gradient_norm = max_gradient_norm
        self.noise_scale = noise_scale
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.device = device

        self.loss_fn = CrossEntropyLoss()
        self.optimizer = AdamW(self.llm.parameters(), lr=5e-4)
        self.freeze_model_layers()
        
        self.total_parameter = 0
        self.privacy_engine = PrivacyEngine()

        
    def freeze_model_layers (self) :
        for p in self.llm.parameters():
            p.requires_grad = False
        
    
    
    def test (self, test_dataloader):
        self.llm.train()
        llm_model, optimizer, train_dataloader = self.privacy_engine.make_private_with_epsilon( 
            module=self.llm, optimizer=self.optimizer, data_loader=test_dataloader,
            target_delta=self.delta, target_epsilon=self.epsilon, epochs=1,
            max_grad_norm=self.max_gradient_norm)
        losses=[]
        accuracies=[]
        with BatchMemoryManager(
            data_loader=test_dataloader, 
            max_physical_batch_size=256, 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for step, batch in enumerate(tqdm(memory_safe_data_loader)):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels':         batch[3]}
                
                while True:
                    pt = self._promptDPSGD(inputs)
                
                    lp_output = llm_model(inputs['input_ids'], [pt])[0]#.detach()
                
                    eps = self.privacy_engine.get_epsilon(self.delta)
                    print(eps)
                    if eps < self.epsilon:
                        print('ok')
                        break

                losses.append(self.loss_fn(lp_output, inputs['labels']))
                
                preds = np.argmax(lp_output.detach().cpu().numpy(), axis=1)
                labels = inputs['labels'].detach().cpu().numpy()

                accuracy = (preds == labels).mean()
                accuracies.append(accuracy)
                

        
    def _promptDPSGD (self, inputs):
        pt = torch.rand((1, self.sequence_length, self.embedding_dim), requires_grad=True)
        for t in range(self.training_iteration):
            #print(len(inputs[list(inputs.keys())[0]]))
            Bt_index = random.sample(range(0, len(inputs[list(inputs.keys())[0]])), 
                               int(self.sampling_rate*len(inputs[list(inputs.keys())[0]])))
            gts=[]
            for i in Bt_index:
                #torch.tensor([inputs['input_ids']])
                lp_output = self.llm(inputs['input_ids'][i].unsqueeze(0), 
                                     [pt])[0].unsqueeze(0)#.detach()
                #lp_output.requires_grad = True
                
                loss = self.loss_fn(lp_output, inputs['labels'][i].unsqueeze(0))
                gt_xi = torch.autograd.grad(loss, pt, allow_unused=True)[0]
                gt_xi_clip = gt_xi/max(1, torch.norm(gt_xi, p=2)/self.max_gradient_norm)
                gts.append(gt_xi_clip)
            gt_hat = (1/len(Bt_index))*(sum(gts)+torch.normal(
                0, self.noise_scale*self.max_gradient_norm*torch.eye(
                    self.sequence_length, self.embedding_dim)).unsqueeze(0))
            pt=pt-(self.lr*gt_hat)
        
        return pt
    
    def train (self, epochs, train_dataloader):
        return
            
        
        