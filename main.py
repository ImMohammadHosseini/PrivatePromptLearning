
import optparse
import torch
import numpy as np
from dataWorkload.sst2Workload import sst2
from src.promptDPSGD import PromptDPSGD
from transformers import AutoModelForSequenceClassification

usage = "usage: python main.py -D <dataset> -M <LLM> -E <epsilon>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-M", "--method", action="store", dest="method", 
                  default='Soft-Prompt',
                  help="variation is Soft-Prompt,")
parser.add_option("-D", "--dataset", action="store", dest="dataset", 
                  default='sst2',
                  help="variation is sst2")
parser.add_option("-L", "--llm", action="store", dest="llm", default='bert-tiny')
parser.add_option("-E", "--epsilon", action="store", dest="epsilon", default=8)

opts, args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#TRAIN = True
BATCH_SIZE = 1024
P_LENGTH = 50
LR = 0.001
NOISE_SCALE = 4
SAMPLING_RATE = 0.01
MAX_GRADIENT_NORM = 4
TRAINING_ITERATION = 180

def init (dataset_name, model_name, method_name, epsilon):
    
    dataWorkload = eval(dataset_name)(model_name)
    llm = AutoModelForSequenceClassification.from_pretrained("prajjwal1/"+ model_name)
    print(llm)
    prompt_learning = PromptDPSGD(method_name, epsilon, llm, TRAINING_ITERATION, 
                     MAX_GRADIENT_NORM, dataWorkload.sequence_length, 
                     llm.bert.embeddings.word_embeddings.weight.size(1),
                     )
    return dataWorkload, llm, prompt_learning
if __name__ == '__main__':
    workload, llm = init(opts.dataset, opts.llm, opts.method, opts.epsilon)