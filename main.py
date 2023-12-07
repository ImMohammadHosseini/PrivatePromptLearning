
import optparse
import torch
import math
import numpy as np
from dataWorkload.sst2Workload import sst2
from dp_learning.soft_prompt import Soft_Prompt
from dp_learning.full_finetune import Full_Fintune
from src.llm_format import LLM_Format

usage = "usage: python main.py -D <dataset> -M <LLM> -E <epsilon>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-M", "--method", action="store", dest="method", 
                  default='Soft_Prompt',
                  help="variation is Soft_Prompt, Prefix, LoRA, Full_Fintune, Last_Layer_Finetune")
parser.add_option("-D", "--dataset", action="store", dest="dataset", 
                  default='sst2',
                  help="variation is sst2")
parser.add_option("-L", "--llm", action="store", dest="llm", default='bert-tiny')
parser.add_option("-E", "--epsilon", action="store", dest="epsilon", default=8)
parser.add_option("-T", "--delta", action="store", dest="delta", default=1e-5)

opts, args = parser.parse_args()

DEVICE = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")


#TRAIN = True
EPOCH = 10 #if opts.method !=
BATCH_SIZE = 64
P_LENGTH = 50
LR = 0.001
NOISE_SCALE = math.sqrt(2*math.log2(1.25/opts.delta))/opts.epsilon
SAMPLING_RATE = 0.01
MAX_GRADIENT_NORM = 4
TRAINING_ITERATION = 180

def init (dataset_name, model_name, method_name):
    
    dataWorkload = eval(dataset_name)()
    llm = LLM_Format(model_name)
    #print(llm)
    prompt_learning = eval(method_name)(llm, opts.epsilon, opts.delta,
                                        SAMPLING_RATE, TRAINING_ITERATION, 
                                        MAX_GRADIENT_NORM, NOISE_SCALE, 
                                        dataWorkload.sequence_length, 
                                        llm.bert.embeddings.word_embeddings.weight.size(1), 
                                        LR, DEVICE)
    return dataWorkload, prompt_learning
if __name__ == '__main__':
    workload, prompt_learning = init(opts.dataset, opts.llm, opts.method)
    prompt_learning.train(EPOCH, workload.get_dataloader(BATCH_SIZE, set_type='train'))
    prompt_learning.test(workload.get_dataloader(BATCH_SIZE, set_type='test'))
    
    
    
    
    
    