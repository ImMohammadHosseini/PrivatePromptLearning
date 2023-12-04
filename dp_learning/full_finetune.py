
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import tqdm

class Soft_Prompt():
    def __init__(self, llm, epsilon, delta, sampling_rate, training_iteration, 
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
        self.freeze_model_layers()
        
        self.total_parameter = 0
        self.optimizer = AdamW(self.llm.parameters(), lr=5e-4)
        self.privacy_engine = PrivacyEngine()
        
    def freeze_model_layers (self) :
        for p in self.llm.parameters():
            p.requires_grad = True
            self.total_parameter += p.numel()
            
    def train (self, epochs, train_dataloader):
        llm_model, optimizer, train_dataloader = self.privacy_engine.make_private_with_epsilon( 
            module=self.model, optimizer=self.optimizer, data_loader=train_dataloader,
            target_delta=self.delta, target_epsilon=self.epsilon, epochs=epochs,
            max_grad_norm=self.max_gradient_norm)
        
        for epoch in range(1, epochs+1):
            losses = []
            accuracies = []
            with BatchMemoryManager(
                data_loader=train_dataloader, 
                max_physical_batch_size=256, 
                optimizer=optimizer
            ) as memory_safe_data_loader:
                for step, batch in enumerate(tqdm(memory_safe_data_loader)):
                    optimizer.zero_grad()
        
                    batch = tuple(t.to(self.device) for t in batch)
                    inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels':         batch[3]}
        
                    outputs = llm_model(**inputs) # output = loss, logits, hidden_states, attentions
                    
                    preds = np.argmax(outputs[:2].detach().cpu().numpy(), axis=1)
                    labels = inputs['labels'].detach().cpu().numpy()
                    accuracy = (preds == labels).mean()
                    accuracies.append(accuracy)
                    
                    loss = outputs[0]
                    loss.backward()
                    losses.append(loss.item())
        
                    optimizer.step()
        
                    if step > 0 and step % 5000 == 0:
                        train_loss = np.mean(losses)
                        train_accuracy = np.mean(accuracies)
                        eps = self.privacy_engine.get_epsilon(self.delta)
        
                        #llm_model.train()
                        #eval_loss, eval_accuracy = evaluate(model)
        
                        print(
                          f"Epoch: {epoch} | "
                          f"Step: {step} | "
                          f"Train loss: {train_loss:.3f} | "
                          f"Train accuracy: {train_accuracy:.3f} | "
                          f"ɛ: {eps:.2f}"
                        )

    def test (self, test_dataloader):
        