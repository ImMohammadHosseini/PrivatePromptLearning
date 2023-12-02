
from os import path, makedirs, listdir, remove
import wget
import shutil
import pandas as pd
from zipfile import ZipFile
from datasets import Dataset

from transformers import AutoTokenizer

class sst2():
    def __init__(self, model_name):
        dataset_path = 'dataWorkload/datasets/SST-2/'
        if not path.exists(dataset_path):
            #makedirs(dataset_path)
            print('Downloading sst2 Dataset')
            
            url = 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip'
            filename = wget.download(url)
            
            zf = ZipFile(filename, 'r'); zf.extractall(dataset_path); zf.close()
            remove(filename)        
        self.dataset_path = dataset_path+'SST-2/'
        train_data = pd.read_csv(self.dataset_path+'train.tsv', sep='\t')
        
        self.sequence_length = max([len(train_data.iloc[i][0].split(' ')) 
                                    for i in range(len(train_data))])
        
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/"+ model_name)
        
        
        
        #label_train = train_data[train_data.columns.values[1]].values.tolist() 
        #label2hot = {
        #    0: [1,0],
        #    1: [0,1]
        #    }
        #train_data[train_data.columns.values[1]] = [
        #    label2hot[l] for l in label_train]
        self.train_dataset = Dataset.from_dict(train_data).with_format("torch")
        
        self.train_dataset = self.train_dataset.map(self.tokenize_function)
        
    def tokenize_function(self, examples):
        return self.tokenizer(examples["sentence"], truncation=True, 
                         padding="max_length", max_length=self.sequence_length)
    
   
    